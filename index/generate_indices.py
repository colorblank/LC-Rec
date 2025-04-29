import argparse
import collections
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EmbDataset
from models.rqvae import RQVAE


def check_collision(all_indices_str):
    """
    检查是否存在冲突的索引。

    参数:
        all_indices_str (list): 索引字符串列表。

    返回:
        bool: 如果没有冲突返回 True，否则返回 False。
    """
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    """
    获取每个索引出现的次数。

    参数:
        all_indices_str (list): 索引字符串列表。

    返回:
        dict: 每个索引及其出现次数的字典。
    """
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    """
    获取发生冲突的索引项。

    参数:
        all_indices_str (list): 索引字符串列表。

    返回:
        list: 冲突索引项的列表。
    """
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def generate_code(prefix, index):
    """
    根据前缀和索引生成编码。

    参数:
        prefix (list): 前缀列表。
        index (list): 索引列表。

    返回:
        list: 生成的编码列表。
    """
    code = []
    for i, ind in enumerate(index):
        code.append(prefix[i].format(int(ind)))
    return code


def save_indices(output_file, all_indices):
    """
    将所有索引保存到文件。

    参数:
        output_file (str): 输出文件路径。
        all_indices (list): 所有索引列表。
    """
    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    with open(output_file, "w") as fp:
        json.dump(all_indices_dict, fp)


def main():
    parser = argparse.ArgumentParser(description="Generate indices")
    parser.add_argument("--dataset", type=str, default="Games", help="Dataset name")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    dataset = args.dataset
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    output_file = f"{dataset}.index.json"
    output_file = os.path.join(output_dir, output_file)
    device = torch.device(args.device)

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(args.data_path)

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = generate_code(prefix, index)

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while True:
        if tt >= 20 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)

            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = generate_code(prefix, index)

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    save_indices(output_file, all_indices)


if __name__ == "__main__":
    main()
