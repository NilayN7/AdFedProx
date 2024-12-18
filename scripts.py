import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def loss_classifier(predictions, labels):
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction="mean")

    return loss(m(predictions), labels.view(-1))


def loss_dataset(model, dataset, loss_f):
    """Compute the loss of `model` on `dataset`"""
    loss = 0

    for idx, (features, labels) in enumerate(dataset):
        predictions = model(features)
        loss += loss_f(predictions, labels)

    loss /= idx + 1
    return loss


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `dataset`"""

    correct = 0

    for features, labels in iter(dataset):
        predictions = model(features)

        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy


def train_step(model, model_0, mu: int, optimizer, train_data, loss_f, q):
    """Train `model` on one epoch of `train_data`"""

    total_loss = 0

    for idx, (features, labels) in enumerate(train_data):
        optimizer.zero_grad()

        predictions = model(features)

        loss = loss_f(predictions, labels)
        loss += mu / 2 * difference_models_norm_2(model, model_0)
        total_loss += (1/(q+1)) * loss ** (1 + q)

        loss.backward()
        optimizer.step()

    return total_loss / (idx + 1)


def local_learning(model, mu: float, optimizer, train_data, epochs: int, loss_f, q):
    model_0 = deepcopy(model)

    for e in range(epochs):
        local_loss = train_step(model, model_0, mu, optimizer, train_data, loss_f, q) 

    return float(local_loss.detach().numpy())


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum([torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
                for i in range(len(tensor_1))])

    return norm


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)
