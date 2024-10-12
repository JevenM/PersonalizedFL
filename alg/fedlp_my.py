import torch
import torch.nn as nn
import torch.optim as optim

from util.modelsel import modelsel
from util.traineval import train, train_prompt
from alg.core.comm import communication, communication_prompt

from alg.fedavg import fedavg

class fedlp(fedavg):
    def __init__(self, args):
        super(fedlp, self).__init__(args)

    def client_train(self, c_idx, dataloader, round):
        if round < 20:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train_prompt(
                self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_aggre(self, round):
        if round < 20:
            self.server_model, self.client_model = communication(
                self.args, self.server_model, self.client_model, self.client_weight)
        else:
            self.server_model, self.client_model = communication_prompt(
                self.args, self.server_prompt, self.client_prompt, self.client_weight)

