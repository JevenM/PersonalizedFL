# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import copy


def communication(args, server_model, models, client_weights):
    client_num = len(models)
    with torch.no_grad():
        if args.alg.lower() == "fedbn":
            for key in server_model.state_dict().keys():
                if "bn" not in key:
                    temp = torch.zeros_like(
                        server_model.state_dict()[key], dtype=torch.float32
                    )
                    for client_idx in range(client_num):
                        temp += (
                            client_weights[client_idx]
                            * models[client_idx].state_dict()[key]
                        )
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )
        elif args.alg.lower() == "fedap":
            tmpmodels = []
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(
                            server_model.state_dict()[key], dtype=torch.float32
                        )
                        for client_idx in range(client_num):
                            temp += (
                                client_weights[cl, client_idx]
                                * tmpmodels[client_idx].state_dict()[key]
                            )
                        server_model.state_dict()[key].data.copy_(temp)
                        if "bn" not in key:
                            models[cl].state_dict()[key].data.copy_(
                                server_model.state_dict()[key]
                            )
        else:
            for key in server_model.state_dict().keys():
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(
                        models[0].state_dict()[key]
                    )
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += (
                            client_weights[client_idx]
                            * models[client_idx].state_dict()[key]
                        )
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )
    return server_model, models


def communication_prompt(args, server_model, client_models, client_weights):
    # with torch.no_grad():
    #     for key in server_model.state_dict().keys():
    #         if 'prompt' == key:
    #             print(f"keeeeeeeeeeeey: {key}")
    #             print(f"before: {server_model.state_dict()[key]}")
    #             temp = torch.zeros_like(server_model.state_dict()[key])
    #             print(temp.shape)
    #             for client_idx in range(len(client_weights)):
    #                 temp += client_weights[client_idx] * client_models[client_idx].state_dict()[key]
    #             print(f"temp: {temp}")
    #             server_model.state_dict()[key].data.copy_(temp)
    #             print(f"after: {server_model.state_dict()[key]}, {server_model.prompt}")
    #             for client_idx in range(len(client_weights)):
    #                 client_models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    #         break
    with torch.no_grad():
        # print(f"before: {server_model.prompt}")
        temp = torch.zeros_like(server_model.prompt)

        # print(temp, temp.shape)
        # 默认是平均，每个权值都是0.05
        for client_idx in range(len(client_weights)):
            temp += client_weights[client_idx] * svd_dcomp(
                client_models[client_idx].prompt
            )
        server_model.prompt.copy_(temp)
        server_model.prompt.requires_grad = False
        # print(f"after: {server_model.prompt}")

        for client_idx in range(len(client_weights)):
            client_models[client_idx].prompt.copy_(temp)
            client_models[client_idx].prompt.requires_grad = True
    return server_model, client_models


def svd_dcomp(ppp):
    # 1. 计算 SVD 分解
    U, S, Vt = torch.linalg.svd(ppp, full_matrices=False)  # 使用torch.linalg.svd计算SVD
    # print(S.shape)
    # 计算累计贡献率
    explained_variance = torch.cumsum(S, dim=0) / torch.sum(S)
    # print("累计贡献率:", explained_variance)

    # 设置阈值 (如 90%)
    threshold = 0.9
    k = torch.where(explained_variance >= threshold)[0][0].item() + 1
    # print(f"选择的 k 值: {k}")
    S_k = S[:k]  # 选择前 k 个奇异值
    U_k = U[:, :k]  # U 的前 k 列
    Vt_k = Vt[:k, :]  # V^T 的前 k 行

    # 构造 Σ_k 矩阵
    S_k_matrix = torch.diag(S_k)

    # 3. 重构矩阵 A'
    ppp_approx = U_k @ S_k_matrix @ Vt_k  # 重构矩阵
    return ppp_approx


if __name__ == "__main__":
    # 定义原始矩阵 A
    ppp = torch.tensor([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]], dtype=torch.float32)

    svd_dcomp(ppp)
