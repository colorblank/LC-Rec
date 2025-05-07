import argparse
import os
import random
import h5py
import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils.common_utils import load_json, set_device, load_plm
from utils.text_utils import clean_text


def load_data(args: argparse.Namespace) -> dict:
    """加载商品特征数据。

    从指定路径加载商品特征的 JSON 文件。

    Args:
        args: 包含数据路径和数据集名称的参数对象。
            - root: 数据根目录
            - dataset: 数据集名称

    Returns:
        dict: 商品ID到商品特征的映射字典。
    """
    item2feature_path = os.path.join(args.root, f"{args.dataset}.item.json")
    item2feature = load_json(item2feature_path)

    return item2feature


def generate_text(item2feature: dict, features: list[str]) -> list[list]:
    """生成商品文本列表。

    从商品特征中提取指定字段的文本内容，并进行清洗。

    Args:
        item2feature: 商品ID到商品特征的映射字典。
        features: 需要提取的特征字段列表，如['title', 'description']。

    Returns:
        list[list]: 商品文本列表，每个元素为 [item_id, [text1, text2, ...]]。
    """
    item_text_list = []

    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())

        item_text_list.append([str(item), text])

    return item_text_list


def preprocess_text(args: argparse.Namespace) -> list[list]:
    """预处理商品文本数据。

    加载并处理商品文本数据，包括标题和描述信息。

    Args:
        args: 包含数据处理相关参数的对象。
            - dataset: 数据集名称

    Returns:
        list[list]: 处理后的商品文本列表，每个元素为 [item_id, [title, description]]。
    """
    print("Process text data: ")
    print(" Dataset: ", args.dataset)

    item2feature = load_data(args)
    # load item text and clean
    item_text_list = generate_text(item2feature, ["title", "description"])
    # item_text_list = generate_text(item2feature, ['title'])
    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def _apply_word_dropout(sentence: str, word_drop_ratio: float) -> str:
    """对句子应用词丢弃。"""
    if word_drop_ratio <= 0:
        return sentence
    words = sentence.split(" ")
    kept_words = [wd for wd in words if random.random() > word_drop_ratio]
    return " ".join(kept_words)


def _get_sentence_embedding(
    sentence: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    args: argparse.Namespace,
) -> torch.Tensor:
    """生成单个句子的嵌入向量。"""
    encoded_sentence = tokenizer(
        [sentence],  # Tokenizer expects a list of sentences
        max_length=args.max_sent_len,
        truncation=True,
        return_tensors="pt",
        padding="longest",
    ).to(args.device)

    outputs = model(
        input_ids=encoded_sentence.input_ids,
        attention_mask=encoded_sentence.attention_mask,
    )

    masked_output = outputs.last_hidden_state * encoded_sentence[
        "attention_mask"
    ].unsqueeze(-1)
    mean_output = masked_output.sum(dim=1) / encoded_sentence["attention_mask"].sum(
        dim=-1, keepdim=True
    )
    return mean_output.detach().cpu()


def _get_item_field_embeddings(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    args: argparse.Namespace,
    word_drop_ratio: float,
) -> list[torch.Tensor]:
    """为单个商品的所有文本字段生成嵌入列表。"""
    field_embeddings = []
    for sentence in texts:
        processed_sentence = _apply_word_dropout(sentence, word_drop_ratio)
        if not processed_sentence.strip():  # if sentence becomes empty after dropout
            # print(f"Warning: Sentence for an item became empty after word dropout. Original: '{sentence}'")
            continue
        embedding = _get_sentence_embedding(processed_sentence, tokenizer, model, args)
        field_embeddings.append(embedding)
    return field_embeddings


def generate_item_embedding(
    args: argparse.Namespace,
    item_text_list: list[list],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    word_drop_ratio: float = -1,
) -> None:
    """生成商品文本的嵌入向量。

    使用预训练语言模型对商品文本进行编码，生成嵌入向量。

    Args:
        args: 包含模型参数和数据路径的参数对象。
            - dataset: 数据集名称
            - max_sent_len: 最大句子长度
            - device: 运行设备
            - root: 数据根目录
            - plm_name: 预训练模型名称
        item_text_list: 商品文本列表，每个元素为 [item_id, [text1, text2, ...]]。
        tokenizer: Huggingface tokenizer 对象。
        model: Huggingface 预训练模型对象。
        word_drop_ratio: 词丢弃率，默认为-1表示不进行词丢弃。

    Returns:
        None: 生成的嵌入向量保存到文件中。
    """
    print("Generate Text Embedding: ")
    print(" Dataset: ", args.dataset)

    output_path = os.path.join(
        args.root, args.dataset + ".emb-" + args.plm_name + "-td" + ".h5"
    )

    with h5py.File(output_path, "w") as hf:
        with torch.no_grad():
            for i, (item_id, texts) in enumerate(item_text_list):
                if (i + 1) % 100 == 0:
                    print(f"==> Processed {i + 1}/{len(item_text_list)} items")

                if not texts:
                    print(f"Warning: Item {item_id} has no text features. Skipping.")
                    continue

                field_embeddings = _get_item_field_embeddings(
                    texts, tokenizer, model, args, word_drop_ratio
                )

                if not field_embeddings:
                    print(
                        f"Warning: No embeddings generated for item {item_id} (possibly all texts were empty or became empty after dropout). Skipping."
                    )
                    continue

                item_embedding_avg = torch.stack(field_embeddings, dim=0).mean(dim=0)
                hf.create_dataset(item_id, data=item_embedding_avg.squeeze().numpy())

    print(f"Embeddings saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    设置并解析用于文本嵌入生成的命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Arts", help="Instruments / Arts / Games"
    )
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=2, help="ID of running GPU")
    parser.add_argument("--plm_name", type=str, default="llama")
    parser.add_argument("--plm_checkpoint", type=str, default="")
    parser.add_argument("--max_sent_len", type=int, default=2048)
    parser.add_argument(
        "--word_drop_ratio",
        type=float,
        default=-1,
        help="word drop ratio, do not drop by default",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    args.device = device

    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_model = load_plm(args.plm_checkpoint)
    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0
    plm_model = plm_model.to(device)

    generate_item_embedding(
        args,
        item_text_list,
        plm_tokenizer,
        plm_model,
        word_drop_ratio=args.word_drop_ratio,
    )
