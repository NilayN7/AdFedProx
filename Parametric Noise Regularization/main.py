import torch
import argparse
from server import FedProx


def main(args):
    fedProx = FedProx(args)
    fedProx.server()
    fedProx.global_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # get data args: (get_data fucntino will recieve alll the args and this will need to send all the args into prepare_data)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reweight_groups", action="store_true",
                            default=False,
                            help="set to True if loss_type is group DRO")

    # prepare_Data_args:
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("-s",
                            "--shift_type",
                            default="confounder")

    # prepare_Confounder_Data_Args:
    parser.add_argument("-d",
                            "--dataset",
                            required=True)
    parser.add_argument("-t", "--target_name")						
    parser.add_argument("-c", "--confounder_names", nargs="+")
    parser.add_argument("--model",
                            # choices=model_attributes.keys(),
                            default="resnet50")
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument(
            "--metadata_csv_name",
            type=str,
            default="metadata.csv",
            help="name of the csv data file (dataset csv has to be placed in dataset folder).",
        )
    parser.add_argument("--fraction", type=float, default=1.0)

    # fedProx_setting_args:
    parser.add_argument("--dataset_name", default="cub")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="resnet50")
    parser.add_argument('--K', type=int, default=3, help='number of total clients')
    clients = ['Task1_W_Zone' + str(i) for i in range(1, 11)]
    parser.add_argument('--clients', default=clients)
    parser.add_argument('--r', type=int, default=10, help='number of communication rounds')
    parser.add_argument('--C', type=float, default=0.5, help='sampling rate')


    # training_args:
    # training args should contain the get_data_args
    # as we are calling the get_DAta fucntion in the train function
    parser.add_argument("--generalization_adjustment", default="0.0")
    parser.add_argument("--loss_type", default="erm",
                            choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--wandb", default=None)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--robust_step_size", default=0.01, type=float)
    parser.add_argument("--use_normalized_loss",
                            default=False,
                            action="store_true")
    parser.add_argument("--btl", default=False, action="store_true")
    parser.add_argument("--minimum_variational_weight", type=float, default=0)
    parser.add_argument("--joint_dro_alpha", default=1, type=float,
                            help=("Size param for CVaR joint DRO."
                                " Only used if loss_type is joint_dro"))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0001)    
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=20)
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--save_last", action="store_true", default=True)
    parser.add_argument("--automatic_adjustment",
                            default=False,
                            action="store_true")

    # parser.add_argument('--E', type=int, default=30, help='number of rounds of training')
    # parser.add_argument('--r', type=int, default=20, help='number of communication rounds')
    # parser.add_argument('--K', type=int, default=10, help='number of total clients')
    # parser.add_argument('--input_dim', type=int, default=28, help='input dimension')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--C', type=float, default=0.5, help='sampling rate')
    # parser.add_argument('--B', type=int, default=50, help='local batch size')
    # parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')
    # parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    # parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    # parser.add_argument('--step_size', type=int, default=10, help='step size')
    # parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay per global round')
    # clients = ['Task1_W_Zone' + str(i) for i in range(1, 11)]
    # parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    main(args)