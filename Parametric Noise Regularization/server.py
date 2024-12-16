import os
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

# from model import ANN
from client import train, test
from models.model_wrapper import ModelWrapper
from utils import Logger


class FedProx:
    def __init__(self, args):
        self.args = args
        # name, model_name, num_classes, num_channels=3
        if args.dataset_name == "cub":
            self.nn = ModelWrapper(name='server', model_name=args.model_name, num_classes=2).model.to(args.device)
            print("this is the model: ", self.nn)
        # self.nn = ANN(args=self.args, name='server').to(args.device)
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index)
            # aggregation
            self.aggregation(index)

        return self.nn

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index):  # update nn
        if os.path.exists(self.args.log_dir):
            mode = "a"
        else:
            mode = "w"
        self.logger = Logger(os.path.join(self.args.log_dir, f"log.txt"), mode)

        print("this is the value of index: ",  index)
        print("this is the len of index: ", len(index))
        for k in index:
            print("#######################################################")
            print(f"training {k}-th model")
            print("#######################################################")
            self.nns[k] = train(self.args, self.nns[k], self.logger)

    def global_test(self):
        model = self.nn
        model.eval()
        for client in self.args.clients:
            model.name = client
            test(self.args, model, self.logger)