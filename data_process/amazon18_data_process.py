import argparse
import collections
import gzip
import json
import os

from tqdm import tqdm

from utils import (
    amazon18_dataset2fullname,
    check_path,
    clean_text,
    write_json_file,
    write_remap_index,
)


def load_ratings(
    file: str,
) -> tuple[set[str], set[str], set[tuple[str, str, float, int]]]:
    """从评分文件中加载用户-物品交互数据。

    Args:
        file: 评分数据文件路径，文件格式为CSV，每行包含物品ID、用户ID、评分和时间戳。

    Returns:
        tuple 包含三个集合:
            - users: 用户ID集合
            - items: 物品ID集合
            - inters: 用户-物品交互记录集合，每条记录为(user_id, item_id, rating, timestamp)元组

    Raises:
        ValueError: 当数据行格式不正确时抛出。
    """
    users, items, inters = set(), set(), set()
    with open(file, "r") as fp:
        for line in tqdm(fp, desc="Load ratings"):
            try:
                item, user, rating, time = line.strip().split(",")
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return users, items, inters


def load_meta_items(file: str) -> dict[str, dict[str, str]]:
    """从元数据文件中加载物品的详细信息。

    Args:
        file: 物品元数据文件路径，文件格式为gzip压缩的JSON，每行包含一个物品的详细信息。

    Returns:
        dict: 物品元数据字典，key为物品ID，value为包含以下字段的字典:
            - title: 物品标题
            - description: 物品描述
            - brand: 品牌名称
            - categories: 物品类别，多个类别用逗号分隔
    """
    items = {}
    with gzip.open(file, "r") as fp:
        for line in tqdm(fp, desc="Load metas"):
            data = json.loads(line)
            item = data["asin"]
            title = clean_text(data["title"])

            descriptions = data["description"]
            descriptions = clean_text(descriptions)

            brand = data["brand"].replace("by\n", "").strip()

            categories = data["category"]
            new_categories = []
            for category in categories:
                if "</span>" in category:
                    break
                new_categories.append(category.strip())
            categories = ",".join(new_categories).strip()

            items[item] = {
                "title": title,
                "description": descriptions,
                "brand": brand,
                "categories": categories,
            }
    return items


def load_review_data(args, user2id, item2id):
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    review_file_path = os.path.join(
        args.input_path, "Review", dataset_full_name + ".json.gz"
    )

    reviews = {}

    with gzip.open(review_file_path, "r") as fp:
        for line in tqdm(fp, desc="Load reviews"):
            inter = json.loads(line)
            try:
                user = inter["reviewerID"]
                item = inter["asin"]
                if user in user2id and item in item2id:
                    uid = user2id[user]
                    iid = item2id[item]
                else:
                    continue
                if "reviewText" in inter:
                    review = clean_text(inter["reviewText"])
                else:
                    review = ""
                if "summary" in inter:
                    summary = clean_text(inter["summary"])
                else:
                    summary = ""
                reviews[str((uid, iid))] = {"review": review, "summary": summary}

            except ValueError:
                print(line)

    return reviews


def get_user2count(
    inters: list[tuple[str, str, float, int]],
) -> collections.defaultdict[str, int]:
    """统计每个用户的交互次数。

    Args:
        inters: 用户-物品交互记录列表，每条记录为(user_id, item_id, rating, timestamp)元组。

    Returns:
        collections.defaultdict: 用户交互次数字典，key为用户ID，value为该用户的交互次数。
    """
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(
    inters: list[tuple[str, str, float, int]],
) -> collections.defaultdict[str, int]:
    """统计每个物品被交互的次数。

    Args:
        inters: 用户-物品交互记录列表，每条记录为(user_id, item_id, rating, timestamp)元组。

    Returns:
        collections.defaultdict: 物品交互次数字典，key为物品ID，value为该物品被交互的次数。
    """
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(
    unit2count: dict[str, int], threshold: int
) -> tuple[set[str], int]:
    """根据交互次数阈值筛选候选实体。

    Args:
        unit2count: 实体交互次数字典，key为实体ID，value为交互次数。
        threshold: 交互次数阈值，只保留交互次数大于等于该阈值的实体。

    Returns:
        tuple: 包含两个元素:
            - cans: 满足阈值要求的实体ID集合
            - filtered_count: 被过滤掉的实体数量
    """
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(
    inters: list[tuple[str, str, float, int]],
    can_items: dict[str, dict[str, str]] | None = None,
    user_k_core_threshold: int = 0,
    item_k_core_threshold: int = 0,
) -> list[tuple[str, str, float, int]]:
    """过滤用户-物品交互记录。

    Args:
        inters: 用户-物品交互记录列表，每条记录为(user_id, item_id, rating, timestamp)元组。
        can_items: 候选物品元数据字典，用于过滤没有元数据的物品。
        user_k_core_threshold: 用户k-core过滤阈值，保留交互次数大于等于该阈值的用户。
        item_k_core_threshold: 物品k-core过滤阈值，保留被交互次数大于等于该阈值的物品。

    Returns:
        list: 过滤后的用户-物品交互记录列表。
    """
    new_inters = []

    # filter by meta items
    if can_items:
        print("\nFiltering by meta items: ")
        for unit in inters:
            if unit[1] in can_items.keys():
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print("    The number of inters: ", len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print("\nFiltering by k-core:")
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(  # users is set
                user2count, user_k_core_threshold
            )
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold
            )
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print(
                "    Epoch %d The number of inters: %d, users: %d, items: %d"
                % (idx, len(inters), len(user2count), len(item2count))
            )
    return inters


def make_inters_in_order(
    inters: list[tuple[str, str, float, int]],
) -> list[tuple[str, str, float, int]]:
    """对用户-物品交互记录按时间戳排序并去重。

    对每个用户的交互记录按时间戳升序排序，并去除重复的物品交互，只保留最早的一次交互。

    Args:
        inters: 用户-物品交互记录列表，每条记录为(user_id, item_id, rating, timestamp)元组。

    Returns:
        list: 排序并去重后的用户-物品交互记录列表。
    """
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        interacted_item = set()
        for inter in user_inters:
            if inter[1] in interacted_item:  # 过滤重复交互
                continue
            interacted_item.add(inter[1])
            new_inters.append(inter)
    return new_inters


def preprocess_rating(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, str, float, int]], dict[str, dict[str, str]]]:
    """预处理评分数据，包括加载数据、过滤和排序。

    Args:
        args: 命令行参数对象，包含以下字段:
            - dataset: 数据集名称
            - input_path: 输入数据路径
            - user_k: 用户过滤的k-core阈值
            - item_k: 物品过滤的k-core阈值

    Returns:
        tuple: 包含两个元素:
            - rating_inters: 处理后的交互记录列表，每条记录为(user_id, item_id, rating, timestamp)元组
            - meta_items: 物品元数据字典
    """
    dataset_full_name = amazon18_dataset2fullname[args.dataset]

    print("Process rating data: ")
    print(" Dataset: ", args.dataset)

    # load ratings
    rating_file_path = os.path.join(
        args.input_path, "Ratings", dataset_full_name + ".csv"
    )
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # load item IDs with meta data
    meta_file_path = os.path.join(
        args.input_path, "Metadata", f"meta_{dataset_full_name}.json.gz"
    )
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print("The number of raw inters: ", len(rating_inters))

    rating_inters = make_inters_in_order(rating_inters)

    rating_inters = filter_inters(
        rating_inters,
        can_items=meta_items,
        user_k_core_threshold=args.user_k,
        item_k_core_threshold=args.item_k,
    )

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print("\n")

    return rating_inters, meta_items


def convert_inters2dict(
    inters: list[tuple[str, str, float, int]],
) -> tuple[collections.defaultdict[int, list[int]], dict[str, int], dict[str, int]]:
    """将用户-物品交互记录转换为字典格式。

    将原始的用户-物品交互记录转换为以下格式:
    1. 为用户和物品分配唯一的整数ID
    2. 构建用户交互物品列表字典

    Args:
        inters: 用户-物品交互记录列表，每条记录为(user_id, item_id, rating, timestamp)元组。

    Returns:
        tuple: 包含三个元素:
            - user2items: 用户交互物品列表字典，key为用户ID，value为该用户交互的物品ID列表
            - user2index: 用户ID映射字典，key为原始用户ID，value为分配的整数ID
            - item2index: 物品ID映射字典，key为原始物品ID，value为分配的整数ID
    """
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_data(
    args: argparse.Namespace,
    rating_inters: list[tuple[str, str, float, int]],
) -> tuple[
    collections.defaultdict[int, list[int]],
    dict[int, list[str]],
    dict[int, list[str]],
    dict[int, list[str]],
    dict[str, int],
    dict[str, int],
]:
    """生成训练、验证和测试数据集。

    使用留一法(Leave-one-out)划分数据集:
    1. 将每个用户的最后一次交互作为测试集
    2. 倒数第二次交互作为验证集
    3. 其余交互作为训练集

    Args:
        args: 命令行参数对象，包含数据集名称等信息。
        rating_inters: 用户-物品交互记录列表，每条记录为(user_id, item_id, rating, timestamp)元组。

    Returns:
        tuple: 包含六个元素:
            - user2items: 用户交互物品列表字典，key为用户ID，value为该用户交互的物品ID列表
            - train_inters: 训练集字典，key为用户ID，value为该用户的训练集物品ID列表
            - valid_inters: 验证集字典，key为用户ID，value为该用户的验证集物品ID列表
            - test_inters: 测试集字典，key为用户ID，value为该用户的测试集物品ID列表
            - user2index: 用户ID映射字典，key为原始用户ID，value为分配的整数ID
            - item2index: 物品ID映射字典，key为原始物品ID，value为分配的整数ID
    """
    print("Split dataset: ")
    print(" Dataset: ", args.dataset)

    # generate train valid temp
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert len(user2items[u_index]) == len(train_inters[u_index]) + len(
            valid_inters[u_index]
        ) + len(test_inters[u_index])
    return user2items, train_inters, valid_inters, test_inters, user2index, item2index


def convert_to_atomic_files(args, train_data, valid_data, test_data):
    print("Convert dataset: ")
    print(" Dataset: ", args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.train.inter"), "w"
    ) as file:
        file.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f"{uid}\t{' '.join(seq)}\t{target_item}\n")

    with open(
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.valid.inter"), "w"
    ) as file:
        file.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f"{uid}\t{' '.join(item_seq)}\t{target_item}\n")

    with open(
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.test.inter"), "w"
    ) as file:
        file.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f"{uid}\t{' '.join(item_seq)}\t{target_item}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Arts", help="Instruments / Arts / Games"
    )
    parser.add_argument("--user_k", type=int, default=5, help="user k-core filtering")
    parser.add_argument("--item_k", type=int, default=5, help="item k-core filtering")
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # load interactions from raw rating file
    rating_inters, meta_items = preprocess_rating(args)

    # split train/valid/temp
    all_inters, train_inters, valid_inters, test_inters, user2index, item2index = (
        generate_data(args, rating_inters)
    )

    check_path(os.path.join(args.output_path, args.dataset))

    write_json_file(
        all_inters,
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.inter.json"),
    )
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    item2feature = collections.defaultdict(dict)
    for item, item_id in item2index.items():
        item2feature[item_id] = meta_items[item]

    # reviews = load_review_data(args, user2index, item2index)

    print("user:", len(user2index))
    print("item:", len(item2index))

    write_json_file(
        item2feature,
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.item.json"),
    )
    # write_json_file(reviews, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))

    write_remap_index(
        user2index,
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.user2id"),
    )
    write_remap_index(
        item2index,
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.item2id"),
    )
