import torch
import os
import copy
import torch.nn as nn
import torch.optim as optim
from util.traineval import pretrain_model, test
from util.modelsel import modelsel
from util.traineval import train, train_prompt
from alg.core.comm import communication, communication_prompt

# import clip
import copy
from alg.fedavg import fedavg


class fedlp_clip(fedavg):
    def __init__(self, args):
        super(fedlp_clip, self).__init__(args)

    def set_client_weight(self, train_loaders):
        os.makedirs("./checkpoint/" + "pretrained/", exist_ok=True)
        preckpt = (
            "./checkpoint/"
            + "pretrained/"
            + self.args.dataset
            + "_"
            + str(self.args.batch)
            + "_"
            + str(self.args.pretrained_iters)
            + "_"
            + str(self.args.prompt_dim)
            + "_a"
            + str(self.args.non_iid_alpha)
        )
        self.pretrain_model = copy.deepcopy(self.server_model).to(self.args.device)
        self.server_model.prompt.requires_grad = False
        if not os.path.exists(preckpt):
            pretrain_model(self.args, self.pretrain_model, preckpt, self.args.device)
        else:
            self.pretrain_model.load_state_dict(torch.load(preckpt)["state"])
            print(f"pretrained model train acc: {torch.load(preckpt)['acc']}")

        for client_idx in range(self.args.n_clients):
            self.client_model[client_idx].load_state_dict(
                self.pretrain_model.state_dict()
            )
            for name, param in self.client_model[client_idx].named_parameters():
                if (
                    "conv1" in name
                    or "bn1" in name
                    or "conv2" in name
                    or "bn2" in name
                    or "fc1" in name
                    or "fc2" in name
                    or "fc3" in name
                ):
                    param.requires_grad = False
        self.optimizers = [
            optim.Adam(
                filter(lambda p: p.requires_grad, self.client_model[idx].parameters()),
                lr=self.args.lr,
                weight_decay=self.args.wd,
            )
            for idx in range(self.args.n_clients)
        ]

    def client_train(self, c_idx, dataloader, round, lr_new):
        # 交替训练
        for name, param in self.client_model[c_idx].named_parameters():
            if (
                "conv1" in name
                or "bn1" in name
                or "conv2" in name
                or "bn2" in name
                or "fc1" in name
                or "fc2" in name
                or "fc3" in name
            ):
                param.requires_grad = False
            if "prompt" in name or "fc4" in name or "fc5" in name or "hyp" in name:
                param.requires_grad = True

        self.optimizers[c_idx] = optim.Adam(
            filter(lambda p: p.requires_grad, self.client_model[c_idx].parameters()),
            lr=lr_new,
            weight_decay=self.args.wd,
        )

        if c_idx == 0 and round == 0:
            for name, param in self.client_model[c_idx].named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

            print(f"train prompt: {self.client_model[c_idx]}")

        train_loss, train_acc = train_prompt(
            self.args,
            self.client_model[c_idx],
            self.server_model.prompt,
            dataloader,
            self.optimizers[c_idx],
            self.loss_fun,
            self.args.device,
            1,
        )

        for name, param in self.client_model[c_idx].named_parameters():
            # 第3好，固定classifier，交替prompt和backbone
            # if "prompt" in name or 'fc4' in name or "fc5" in name or 'fc1' in name or "fc2" in name or "fc3" in name:
            #     param.requires_grad = False
            # if "conv1" in name or "bn1" in name or "conv2" in name or "bn2" in name:
            #     param.requires_grad = True

            # 第1好，固定backbone
            if (
                "prompt" in name
                or "fc4" in name
                or "fc5" in name
                or "hyp" in name
                or "conv1" in name
                or "bn1" in name
                or "conv2" in name
                or "bn2" in name
            ):
                param.requires_grad = False
            if "fc1" in name or "fc2" in name or "fc3" in name:
                param.requires_grad = True

            # 第2好，交替训练prompt和classifier+backbone
            # if "prompt" in name or 'fc4' in name or "fc5" in name:
            #     param.requires_grad = False
            # if "conv1" in name or "bn1" in name or "conv2" in name or "bn2" in name or 'fc1' in name or "fc2" in name or "fc3" in name:
            #     param.requires_grad = True
        """
        # 不交替
        for name, param in self.client_model[c_idx].named_parameters():
            if "prompt" in name or 'fc4' in name or "hyp" in name or "fc5" in name:
                print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
        """
        self.optimizers[c_idx] = optim.Adam(
            filter(lambda p: p.requires_grad, self.client_model[c_idx].parameters()),
            lr=lr_new,
            weight_decay=self.args.wd,
        )

        if c_idx == 0 and round == 0:
            for name, param in self.client_model[c_idx].named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

            print(f"train backbone + classifier: {self.client_model[c_idx]}")
        train_loss, train_acc = train_prompt(
            self.args,
            self.client_model[c_idx],
            self.server_model.prompt,
            dataloader,
            self.optimizers[c_idx],
            self.loss_fun,
            self.args.device,
            1,
        )

        return train_loss, train_acc

    def server_aggre(self, client_weight=None):
        if client_weight is not None:
            self.client_weight = client_weight
        self.server_model, self.client_model = communication_prompt(
            self.args, self.server_model, self.client_model, self.client_weight
        )

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx],
            dataloader,
            self.loss_fun,
            self.args.device,
            1,
            self.server_model.prompt,
        )
        return train_loss, train_acc
