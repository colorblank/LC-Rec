import copy

import torch
from transformers import (
    LlamaTokenizer,
)


class Collator(object):
    """数据整理器，用于处理训练数据的批处理和标记化。

    该类负责将输入文本和目标文本进行标记化处理，并生成用于模型训练的批处理数据。
    支持仅训练响应部分的功能，可以通过设置忽略输入文本部分的标签。

    Attributes:
        args: 配置参数对象
        only_train_response: bool, 是否仅训练响应部分
        tokenizer: 分词器对象，用于文本标记化处理
    """

    def __init__(self, args: object, tokenizer: object) -> None:
        """初始化 Collator 对象。

        Args:
            args: 配置参数对象，包含 only_train_response 等训练配置
            tokenizer: 分词器对象，用于文本标记化处理
        """
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    def __call__(self, batch: list) -> dict:
        """处理批量数据。

        将输入的批量数据进行标记化处理，生成模型训练所需的输入数据和标签。

        Args:
            batch: 包含输入文本和标签的批量数据列表

        Returns:
            dict: 包含处理后的输入数据和标签的字典
        """
        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text=full_texts,
            text_target=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels

        return inputs


class TestCollator(object):
    """测试数据整理器，用于处理测试数据的批处理和标记化。

    该类负责将测试输入文本进行标记化处理，并生成用于模型推理的批处理数据。
    支持 LlamaTokenizer 的批量推理功能。

    Attributes:
        args: 配置参数对象
        tokenizer: 分词器对象，用于文本标记化处理
    """

    def __init__(self, args: object, tokenizer: object) -> None:
        """初始化 TestCollator 对象。

        Args:
            args: 配置参数对象
            tokenizer: 分词器对象，用于文本标记化处理
        """
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        if isinstance(self.tokenizer, LlamaTokenizer):
            # Allow batched inference
            self.tokenizer.padding_side = "left"

    def __call__(self, batch: list) -> tuple:
        """处理批量测试数据。

        将输入的批量测试数据进行标记化处理，生成模型推理所需的输入数据和目标。

        Args:
            batch: 包含输入文本和标签的批量数据列表

        Returns:
            tuple: 包含处理后的输入数据和目标的元组
        """
        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)
