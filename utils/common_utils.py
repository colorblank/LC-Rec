"""通用工具模块，包含时间处理、文件操作等基础功能。"""

import datetime
import json
import os


def get_local_time():
    """获取当前时间的格式化字符串。

    Returns:
        str: 格式化的时间字符串，格式为'%b-%d-%Y_%H-%M-%S'。
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    return cur


def ensure_dir(dir_path):
    """确保目录存在，如果不存在则创建。

    Args:
        dir_path: 目录路径。
    """
    os.makedirs(dir_path, exist_ok=True)


def load_json(file):
    """加载JSON文件。

    Args:
        file: JSON文件路径。

    Returns:
        dict: 加载的JSON数据。
    """
    with open(file, "r") as f:
        data = json.load(f)
    return data