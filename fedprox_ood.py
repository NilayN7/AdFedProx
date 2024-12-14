from copy import deepcopy
from scripts import loss_dataset, accuracy_dataset, local_learning, loss_classifier, set_to_zero_model_weights
import torch.optim as optim
import torch.nn as nn
import torch

def average_models(model, clients_models_hist:list , weights:list):


    """Creates the new model of a given iteration with the models of the other
    clients"""
    
    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):
        
        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)
            
    return new_model


def FedProx(model, training_sets:list, n_iter:int, testing_sets:list, mu=0, 
    file_name="test", epochs=5, lr=10**-2, decay=1):
    """ all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the 
            training set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularization term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration
    
    returns :
        - `model`: the final global model 
    """
        
    loss_f=loss_classifier
    
    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=([len(db.dataset)/n_samples for db in training_sets])
    print("Clients' weights:",weights)
    
    
    loss_hist=[[float(loss_dataset(model, dl, loss_f).detach()) 
        for dl in training_sets]]
    acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist=[[tens_param.detach().numpy() 
        for tens_param in list(model.parameters())]]
    models_hist = []
    
    
    server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')
    
    for i in range(n_iter):
        
        clients_params=[]
        clients_models=[]
        clients_losses=[]
        
        for k in range(K):
            
            local_model=deepcopy(model)
            local_optimizer=optim.SGD(local_model.parameters(),lr=lr)
            
            local_loss=local_learning(local_model,mu,local_optimizer,
                training_sets[k],epochs,loss_f)
            
            clients_losses.append(local_loss)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)    
            clients_models.append(deepcopy(local_model))
        
        
        #CREATE THE NEW GLOBAL MODEL
        model = average_models(deepcopy(model), clients_params, 
            weights=weights)
        models_hist.append(clients_models)
        
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach()) 
            for dl in training_sets]]
        acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

        print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
        

        server_hist.append([tens_param.detach().cpu().numpy() 
            for tens_param in list(model.parameters())])
        
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr*=decay
            
    return model, loss_hist, acc_hist


def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1)

def is_ood(energy, threshold):
    pos_energy = -energy
    # print("energy",pos_energy,">>> threshold", threshold)
    return pos_energy <= threshold

def FedEnergy_with_OOD(model, training_sets, n_iter, testing_sets, mu=0, 
                     file_name="test", epochs=5, lr=1e-2, decay=1, energy_threshold=10):
    loss_f = nn.CrossEntropyLoss()
    
    # Variables initialization
    K = len(training_sets)  # number of clients
    n_samples = sum([len(db.dataset) for db in training_sets])
    weights = [len(db.dataset)/n_samples for db in training_sets]
    print("Clients' weights:", weights)
    
    loss_hist = [[float(loss_dataset(model, dl, loss_f).detach()) for dl in training_sets]]
    acc_hist = [[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist = [[tens_param.detach().numpy() for tens_param in list(model.parameters())]]
    models_hist = []
    ood_hist = []  # To keep track of OOD samples detected
    ood_per_hist = []
    server_loss = sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc = sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')
    
    for i in range(n_iter):
        clients_params = []
        clients_models = []
        clients_losses = []
        ood_samples = [0] * K  # Count of OOD samples for each client
        ood_per = [0] * K  # Percentage of OOD samples for each client
        for k in range(K):
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            
            local_loss, ood_count, ood_per_c = local_learning_with_ood(local_model, mu, local_optimizer,
                                                            training_sets[k], epochs, loss_f, energy_threshold)
            
            clients_losses.append(local_loss)
            ood_samples[k] = ood_count
            ood_per[k] = ood_per_c
            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = [tens_param.detach() for tens_param in local_model.parameters()]
            clients_params.append(list_params)    
            clients_models.append(deepcopy(local_model))
        
        # CREATE THE NEW GLOBAL MODEL
        model = average_models(deepcopy(model), clients_params, weights=weights)
        models_hist.append(clients_models)
        
        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist += [[float(loss_dataset(model, dl, loss_f).detach()) for dl in training_sets]]
        acc_hist += [[accuracy_dataset(model, dl) for dl in testing_sets]]
        ood_hist.append(ood_samples)
        ood_per_hist.append(ood_per)

        server_loss = sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc = sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

        print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
        print(f'OOD samples detected: {ood_samples}')
        print(f'OOD percentages: {ood_per}')

        server_hist.append([tens_param.detach().cpu().numpy() for tens_param in list(model.parameters())])
        
        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay
            
    return model, loss_hist, acc_hist, ood_hist, ood_per_hist

def local_learning_with_ood(model, mu, optimizer, train_loader, epochs, loss_f, energy_threshold):
    model.train()
    
    for epoch in range(epochs):
        ood_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate energy scores
            energy = energy_score(output)
            
            # Identify OOD samples
            ood_mask = is_ood(energy, energy_threshold)
            ood_count += ood_mask.sum().item()
            ood_per = ood_count / len(train_loader.dataset)
            
            # Only use in-distribution samples for training
            in_dist_mask = ~ood_mask
            loss = loss_f(output[in_dist_mask], target[in_dist_mask])
            
            # Add proximal term
            if mu != 0:
                loss += mu/2 * proximal_term(model)
                
            loss.backward()
            optimizer.step()
    
    return loss.item(), ood_count, ood_per

def proximal_term(local_model):
    server_model = local_model  # Assuming the initial model is the server model
    prox_term = 0
    for local_param, server_param in zip(local_model.parameters(), server_model.parameters()):
        prox_term += (local_param - server_param).norm(2)**2
    return prox_term

