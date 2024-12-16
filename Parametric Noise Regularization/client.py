import os
import copy

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
# from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from tqdm import tqdm
from get_data import get_data
from loss import LossComputer
from utils import hinge_loss, Logger, log_args, CSVBatchLogger
from data.data import log_data

def run_epoch(
    epoch,
    model,
    optimizer,
    loader,
    loss_computer,
    logger,
    csv_logger,
    args,
    is_training,
    show_progress=False,
    log_every=50,
    scheduler=None,
    csv_name=None,
    wandb_group=None,
    wandb=None,
):
    if is_training:
        model.train()        
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):

        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(args.device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            data_idx = batch[3]

            # outputs.shape: (batch_size, num_classes)
            outputs = model(x)

            output_df = pd.DataFrame()

            # Calculate stats
            if batch_idx == 0:
                acc_y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc_y_true = y.cpu().numpy()
                indices = data_idx.cpu().numpy()
                
                probs = outputs.detach().cpu().numpy()
            else:
                acc_y_pred = np.concatenate([
                    acc_y_pred,
                    np.argmax(outputs.detach().cpu().numpy(), axis=1)
                ])
                acc_y_true = np.concatenate([acc_y_true, y.cpu().numpy()])
                indices = np.concatenate([indices, data_idx.cpu().numpy()])
                probs = np.concatenate([probs, outputs.detach().cpu().numpy()], axis = 0)
                
            assert probs.shape[0] == indices.shape[0]
            # TODO: make this cleaner.
            run_name = f"{csv_name}_epoch_{epoch}_val"
            output_df[f"y_pred_{run_name}"] = acc_y_pred
            output_df[f"y_true_{run_name}"] = acc_y_true
            output_df[f"indices_{run_name}"] = indices
            
            for class_ind in range(probs.shape[1]):
                output_df[f"pred_prob_{run_name}_{class_ind}"] = probs[:, class_ind]

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                run_stats = loss_computer.get_stats(model, args)
                csv_logger.log(epoch, batch_idx, run_stats)

                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                if wandb is not None:
                    wandb_stats = {
                        wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                    }
                    wandb_stats["epoch"] = epoch
                    wandb_stats["batch_idx"] = batch_idx
                    wandb.log(wandb_stats)

        if run_name is not None:
            save_dir = "/".join(csv_logger.path.split("/")[:-1])
            output_df.to_csv(
                os.path.join(save_dir, 
                                f"{model.name}_output_{wandb_group}_epoch_{epoch}.csv"))
            print("Saved", os.path.join(save_dir, 
                                f"{model.name}_output_{wandb_group}_epoch_{epoch}.csv"))


        if (not is_training) or loss_computer.batch_count > 0:
            run_stats = loss_computer.get_stats(model, args)
            if args.wandb is not None:
                assert wandb_group is not None
                wandb_stats = {
                    wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                }
                wandb_stats["epoch"] = epoch
                wandb_stats["batch_idx"] = batch_idx
                wandb.log(wandb_stats)
                print("logged to wandb")

            csv_logger.log(epoch, batch_idx, run_stats)
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()

def train(args, model, logger):
    #################################################################################################################
    # TODO: complete selectign the correct args to be passed in get_data.py and then pass those args in this function
    
    if os.path.exists(args.log_dir):
        mode = "a"
    else:
        mode = "w"

    # Record args
    log_args(args, logger)

    train_data, val_data, test_data, train_loader, val_loader, test_loader = get_data(args)

    data = {}
    data["train_loader"] = train_loader
    data["val_loader"] = val_loader
    data["test_loader"] = test_loader
    data["train_data"] = train_data
    data["val_data"] = val_data
    data["test_data"] = test_data

    log_data(data, logger)
    #################################################################################################################
    model.len = len(train_loader)
    model = model.to(args.device)

    logger.flush()

    csv_name = None

    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"{model.name}_train.csv"),
                                      train_data.n_groups,
                                      mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"{model.name}_val.csv"),
                                    val_data.n_groups,
                                    mode=mode)
    

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, data["train_data"].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * data["train_data"].n_groups)
    else:
        adjustments = np.array(adjustments)

    criterion = hinge_loss

    train_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=data["train_data"],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
        joint_dro_alpha=args.joint_dro_alpha,
    )

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08,
        )
    else:
        scheduler = None
    
    # training
    epoch_offset = 0 
    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        logger.write("\nEpoch [%d]:\n" % epoch)
        logger.write(f"Training:\n")
        run_epoch(
            epoch,
            model,
            optimizer,
            data["train_loader"],
            train_loss_computer,
            logger,
            train_csv_logger,
            args,
            is_training=True,
            csv_name=csv_name,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            wandb_group="train",
            wandb=args.wandb,
        )

        logger.write(f"\nValidation:\n")
        val_loss_computer =  LossComputer(
            criterion,
            loss_type=args.loss_type,
            dataset=data["val_data"],
            alpha=args.alpha,
            gamma=args.gamma,
            adj=adjustments,
            step_size=args.robust_step_size,
            normalize_loss=args.use_normalized_loss,
            btl=args.btl,
            min_var_weight=args.minimum_variational_weight,
            joint_dro_alpha=args.joint_dro_alpha,
        )
        run_epoch(
            epoch,
            model,
            optimizer,
            data["val_loader"],
            val_loss_computer,
            logger,
            val_csv_logger,
            args,
            is_training=False,
            csv_name=csv_name,
            wandb_group="val",
            wandb=args.wandb,
        )

        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                logger.write("Current lr: %f\n" % curr_lr)

        if args.scheduler and args.model != "bert":
            if args.loss_type == "group_dro":
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss,
                    val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(
                val_loss)  # scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir,
                                           "%d_model.pth" % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, f"{model.name}_last_model.pth"))

        if args.save_best:
            if args.loss_type == "group_dro" or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f"Current validation accuracy: {curr_val_acc}\n")
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, f"{model.name}_best_model.pth"))
                best_model = copy.deepcopy(model)
                logger.write(f"Best model saved at epoch {epoch}\n")

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(
                train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write("Adjustments updated\n")
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f"  {train_loss_computer.get_group_name(group_idx)}:\t"
                    f"adj = {train_loss_computer.adj[group_idx]:.3f}\n")
        logger.write("\n")

    train_csv_logger.close()
    val_csv_logger.close()

    return best_model


def test(args, model, logger):
    #################################################################################################################
    # TODO: complete selectign the correct args to be passed in get_data.py and then pass those args in this function
    if os.path.exists(args.log_dir):
        mode = "a"
    else:
        mode = "w"

    # Record args
    log_args(args, logger)

    train_data, val_data, test_data, train_loader, val_loader, test_loader = get_data(args)

    data = {}
    data["train_loader"] = train_loader
    data["val_loader"] = val_loader
    data["test_loader"] = test_loader
    data["train_data"] = train_data
    data["val_data"] = val_data
    data["test_data"] = test_data

    log_data(data, logger)
    #################################################################################################################
    
    criterion = hinge_loss
    epoch = 0 
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    csv_name = None

    test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv"),
                                     test_data.n_groups,
                                     mode=mode)

    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, data["train_data"].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * data["train_data"].n_groups)
    else:
        adjustments = np.array(adjustments)

    test_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=data["test_data"],
        step_size=args.robust_step_size,
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
        joint_dro_alpha=args.joint_dro_alpha,
    )
    run_epoch(
        epoch,
        model,
        optimizer,
        data["test_loader"],
        test_loss_computer,
        logger,
        test_csv_logger,
        args,
        is_training=False,
        csv_name=csv_name,
        wandb_group="test",
        wandb=args.wandb,
    )

    test_csv_logger.close()
