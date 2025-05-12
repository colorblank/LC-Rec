"""通用工具模块，包含项目中使用的通用功能。"""

import collections
import json
import os
import pickle

import torch
from transformers import AutoModel, AutoTokenizer


def ensure_dir(path: str) -> None:
    """检查路径是否存在,不存在则创建。

    Args:
        path: 需要检查的路径。
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id: int) -> torch.device:
    """设置PyTorch计算设备。

    Args:
        gpu_id: GPU设备ID,如果为-1则使用CPU。

    Returns:
        torch.device: PyTorch设备对象。
    """
    if gpu_id == -1:
        return torch.device("cpu")
    else:
        return torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
        )


def load_plm(model_path: str = "bert-base-uncased") -> tuple[AutoTokenizer, AutoModel]:
    """加载预训练语言模型。

    Args:
        model_path: 模型路径,默认为'bert-base-uncased'。

    Returns:
        包含tokenizer和model的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )

    print("Load Model:", model_path)

    model = AutoModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def load_json(file: str) -> dict:
    """加载JSON文件。

    Args:
        file: JSON文件路径。

    Returns:
        加载的JSON数据。
    """
    with open(file, "r") as f:
        data = json.load(f)
    return data


def load_pickle(filename: str) -> any:
    """加载pickle文件。

    Args:
        filename: pickle文件路径。

    Returns:
        加载的pickle数据。
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def make_inters_in_order(inters: list[tuple]) -> list[tuple]:
    """按时间戳对用户交互记录进行排序。

    Args:
        inters: 包含用户交互记录的列表,每条记录为(user, item, rating, timestamp)格式的元组。

    Returns:
        按时间戳排序后的交互记录列表。
    """
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def write_json_file(dic: dict, file: str) -> None:
    """将字典数据写入JSON文件。

    Args:
        dic: 要写入的字典数据。
        file: 输出文件路径。
    """
    print("Writing json file: ", file)
    with open(file, "w") as fp:
        json.dump(dic, fp, indent=4)


def write_remap_index(unit2index: dict, file: str) -> None:
    """将映射索引写入文件。

    Args:
        unit2index: 单元到索引的映射字典。
        file: 输出文件路径。
    """
    print("Writing remap file: ", file)
    with open(file, "w") as fp:
        for unit in unit2index:
            fp.write(unit + "\t" + str(unit2index[unit]) + "\n")


def get_local_time() -> str:
    """获取当前本地时间的字符串表示。"""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
