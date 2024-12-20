# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import argparse
import numpy as np

from datautil.prepare_data import *
from util.config import img_param_init, set_random_seed
from util.evalandprint import evalandprint
from alg import algs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg",
        type=str,
        default="fedavg",
        help="Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed | fedlp]",
    )
    parser.add_argument(
        "--datapercent", type=float, default=1e-1, help="data percent to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pacs",
        help="[vlcs | pacs | officehome | pamap | covid | medmnist]",
    )
    parser.add_argument(
        "--root_dir", type=str, default="/data/mwj/data/", help="data path"
    )
    parser.add_argument(
        "--save_path", type=str, default="./cks/", help="path to save the checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="[cuda | cpu]")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument(
        "--iters", type=int, default=300, help="iterations for communication"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n_clients", type=int, default=20, help="number of clients")
    parser.add_argument(
        "--non_iid_alpha", type=float, default=0.1, help="data split for label shift"
    )
    parser.add_argument(
        "--partition_data",
        type=str,
        default="non_iid_dirichlet",
        help="partition data way",
    )
    parser.add_argument("--plan", type=int, default=1, help="choose the feature type")
    parser.add_argument("--prompt_dim", type=int, default=8, help="")
    parser.add_argument("--wd", type=float, default=0)

    parser.add_argument(
        "--pretrained_iters",
        type=int,
        default=150,
        help="iterations for pretrained models",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--wk_iters",
        type=int,
        default=1,
        help="optimization iters in local worker between communication",
    )
    parser.add_argument("--nosharebn", action="store_true", help="not share bn")

    # algorithm-specific parameters
    parser.add_argument(
        "--mu", type=float, default=1e-3, help="The hyper parameter for fedprox"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="threshold to use copy or distillation, hyperparmeter for metafed",
    )
    parser.add_argument(
        "--lam", type=float, default=1.0, help="init lam, hyperparmeter for metafed"
    )
    parser.add_argument(
        "--model_momentum", type=float, default=0.5, help="hyperparameter for fedap"
    )
    args = parser.parse_args()

    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    if args.dataset in ["vlcs", "pacs", "off_home"]:
        args = img_param_init(args)
        args.n_clients = 4

    exp_folder = f"fed_{args.dataset}_{args.alg}_{args.datapercent}_{args.non_iid_alpha}_{args.mu}_{args.model_momentum}_{args.plan}_{args.lam}_{args.threshold}_{args.iters}_{args.wk_iters}_{args.lr}_{args.prompt_dim}"
    if args.nosharebn:
        exp_folder += "_nosharebn"
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, args.alg + ".pt")

    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)

    algclass = algs.get_algorithm_class(args.alg)(args)

    if args.alg == "fedap" or args.alg == "fedlp":
        print("pretraining...")
        algclass.set_client_weight(train_loaders)
    elif args.alg == "metafed":
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters - 1
        print("Common knowledge accumulation stage")

    best_changed = False

    best_vacc = [0] * args.n_clients
    best_tacc = [0] * args.n_clients
    start_iter = 0
    best_epoch = -1
    val_acc_list = [0 for i_ in range(args.n_clients)]
    lr_new = args.lr
    tau = 10
    # False  # 0.3*args.iters # 总共300的时候选80，什么时候开始不按照平均聚合
    start_agg_round = 90

    for a_iter in range(start_iter, args.iters):
        print(f"============ Train round {a_iter} ============")

        if args.alg == "metafed":
            for c_idx in range(args.n_clients):
                algclass.client_train(
                    c_idx, train_loaders[algclass.csort[c_idx]], a_iter
                )
            algclass.update_flag(val_loaders)
        else:
            if args.alg == "fedlp":
                # if a_iter != 0 and a_iter % 30 == 0:
                #     lr_new = args.lr * (0.99**int(a_iter/30))
                print(f"now global iter: {a_iter} lr: {lr_new}")
                # local client training
                for wi in range(args.wk_iters):
                    for client_idx in range(args.n_clients):
                        algclass.client_train(
                            client_idx, train_loaders[client_idx], a_iter, lr_new
                        )

                if a_iter < start_agg_round:  # not start_agg_round:
                    # server aggregation
                    algclass.server_aggre()
                else:
                    assert len(val_acc_list) != 0
                    "should have values!"
                    # 1反比例权重计算
                    # epsilon = 1e-6  # 防止除零
                    # weight_list = 1 / (np.array(val_acc_list) + epsilon)
                    # # 对权重进行归一化，使其总和为1
                    # weight_list = weight_list / np.sum(weight_list)

                    # 2指数变换权重计算
                    weight_list = np.exp((np.array(val_acc_list)) / tau)
                    # 对权重进行归一化，使其总和为1
                    weight_list = weight_list / np.sum(weight_list)
                    print(f"weight_list: {weight_list}")
                    algclass.server_aggre(weight_list)

                # algclass.server_aggre()
            else:
                # local client training
                for wi in range(args.wk_iters):
                    for client_idx in range(args.n_clients):
                        algclass.client_train(
                            client_idx, train_loaders[client_idx], a_iter
                        )
                algclass.server_aggre()

        best_vacc, best_tacc, best_changed, best_epoch, val_acc_list, train_acc_list = (
            evalandprint(
                args,
                algclass,
                train_loaders,
                val_loaders,
                test_loaders,
                SAVE_PATH,
                best_vacc,
                best_tacc,
                a_iter,
                best_changed,
                best_epoch,
            )
        )

        # 确定什么时候开始
        # if not start_agg_round and np.mean(np.array(train_acc_list)) >= 0.9:
        #     start_agg_round = True
        #     print(f"agg start at: {a_iter}")
    if args.alg == "metafed":
        print("Personalization stage")
        for c_idx in range(args.n_clients):
            algclass.personalization(
                c_idx,
                train_loaders[algclass.csort[c_idx]],
                val_loaders[algclass.csort[c_idx]],
            )
        best_vacc, best_tacc, best_changed, best_epoch, _, _ = evalandprint(
            args,
            algclass,
            train_loaders,
            val_loaders,
            test_loaders,
            SAVE_PATH,
            best_vacc,
            best_tacc,
            a_iter,
            best_changed,
            best_epoch,
        )

    s = "Personalized test acc for each client: "
    for item in best_tacc:
        s += f"{item:.4f},"
    mean_acc_test = np.mean(np.array(best_tacc))
    s += f"\nAverage accuracy: {mean_acc_test:.4f}, best_epoch: {best_epoch}"
    print(s)
