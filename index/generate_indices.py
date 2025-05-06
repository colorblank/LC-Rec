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

    # 保存索引元素到item_id的映射
    index_to_item_id = {}
    for item_id, indices in all_indices_dict.items():
        for index in indices:
            index_to_item_id.setdefault(index, []).append(item_id)

    # 保存索引元素到item_id的映射到文件
    index_to_item_id_file = output_file.replace(".index.json", ".index_to_item_id.json")
    with open(index_to_item_id_file, "w") as fp:
        json.dump(index_to_item_id, fp)

    with open(output_file, "w") as fp:
        json.dump(all_indices_dict, fp)


def main():
    """
    主函数，用于生成数据集的索引并处理模型检查点。
    解析命令行参数，加载数据集和模型，生成并优化索引，最后保存索引。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Generate indices")
    # 添加命令行参数
    parser.add_argument("--dataset", type=str, default="Games", help="Dataset name")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    # 解析命令行参数
    args = parser.parse_args()

    # 从解析的参数中提取变量
    dataset = args.dataset
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    # 构造输出文件名
    output_file = f"{dataset}.index.json"
    # 拼接输出文件路径
    output_file = os.path.join(output_dir, output_file)
    # 根据参数选择设备
    device = torch.device(args.device)

    # 加载模型检查点
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    # 从检查点中提取参数和状态字典
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # 根据数据集路径加载数据
    data = EmbDataset(args.data_path)

    # 初始化模型
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

    # 加载模型权重
    model.load_state_dict(state_dict)
    # 将模型转移到指定设备
    model = model.to(device)
    # 设置模型为评估模式
    model.eval()
    # 打印模型结构

    # 创建数据加载器
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
    )

    # 初始化索引列表
    all_indices = []
    all_indices_str = []
    # 定义索引前缀
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    # 使用模型生成初始索引
    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = generate_code(prefix, index)

            all_indices.append(code)
            all_indices_str.append(str(code))

    # 将索引列表转换为数组
    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # 调整Softmax-Kmeans的参数以优化索引
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    # 初始化迭代计数器
    tt = 0
    # 通过迭代解决索引冲突
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

    # 打印索引统计信息
    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    # 保存最终的索引
    save_indices(output_file, all_indices)


if __name__ == "__main__":
    main()
