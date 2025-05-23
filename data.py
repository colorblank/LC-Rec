import copy
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset

from prompt import all_prompt, sft_prompt


class EmbDataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class BaseDataset(Dataset):
    """数据集的基类，提供通用的数据加载和处理功能。

    Attributes:
        args: 命令行参数。
        dataset (str): 数据集名称。
        data_path (str): 数据集路径。
        max_his_len (int): 历史序列的最大长度。
        his_sep (str): 历史序列的分隔符。
        index_file (str): 索引文件名。
        add_prefix (bool): 是否为历史序列添加前缀。
        new_tokens (Optional[List[str]]): 数据集中所有新的 token。
        allowed_tokens (Optional[Dict[int, Set[int]]]): 允许的 token ID 字典。
        all_items (Optional[Set[str]]): 数据集中所有的 item。
        indices (Dict[str, List[str]]): item ID 到索引 token 的映射。
    """

    def __init__(self, args: Any):
        """初始化 BaseDataset。

        Args:
            args: 包含数据集配置的参数对象。
        """
        super().__init__()

        self.args = args
        self.dataset: str = args.dataset
        self.data_path: str = os.path.join(args.data_path, self.dataset)

        self.max_his_len: int = args.max_his_len
        self.his_sep: str = args.his_sep
        self.index_file: str = args.index_file
        self.add_prefix: bool = args.add_prefix

        self.new_tokens: Optional[List[str]] = None
        self.allowed_tokens: Optional[Dict[int, Set[int]]] = None
        self.all_items: Optional[Set[str]] = None
        self.indices: Dict[str, List[str]] = {}

    def _load_data(self) -> None:
        """从索引文件加载 item 索引数据。"""
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)

    def get_new_tokens(self) -> List[str]:
        """获取数据集中所有唯一的 token，并排序。

        Returns:
            List[str]: 排序后的唯一 token 列表。
        """
        if self.new_tokens is not None:
            return self.new_tokens

        new_tokens_set: Set[str] = set()
        for index in self.indices.values():
            for token in index:
                new_tokens_set.add(token)
        self.new_tokens = sorted(list(new_tokens_set))

        return self.new_tokens

    def get_all_items(self) -> Set[str]:
        """获取数据集中所有 item 的字符串表示。

        Returns:
            Set[str]: 所有 item 的集合。
        """
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(
        self, tokenizer: Any
    ) -> Callable[[int, List[int]], List[int]]:
        """获取一个用于限制生成 token 的前缀函数。

        Args:
            tokenizer: 用于编码 token 的分词器。

        Returns:
            Callable[[int, List[int]], List[int]]: 一个函数，根据当前生成的序列返回允许的下一个 token ID 列表。
        """
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    # 假设 tokenizer 返回的 input_ids 第一个元素是特殊 token，取第二个
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens:
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            # 添加 EOS token
            self.allowed_tokens[len(self.allowed_tokens)] = {tokenizer.eos_token_id}

        # 假设 tokenizer 返回的 input_ids 第一个元素是特殊 token，取后面的
        sep: List[int] = tokenizer("Response:")["input_ids"][1:]
        sep_len = len(sep)
        sep_reversed = sep[::-1]

        def prefix_allowed_tokens_fn(batch_id: int, sentence: List[int]) -> List[int]:
            """根据当前生成的句子，确定下一个允许的 token ID。

            Args:
                batch_id (int): 当前批次的 ID。
                sentence (List[int]): 当前已生成的 token ID 列表。

            Returns:
                List[int]: 允许的下一个 token ID 列表。
            """
            # sentence = sentence.tolist() # sentence 已经是 list
            reversed_sent = sentence[::-1]
            sent_len = len(reversed_sent)
            for i in range(sent_len):
                # 检查是否匹配到 'Response:' 分隔符的反向序列
                if (
                    i + sep_len <= sent_len
                    and reversed_sent[i : i + sep_len] == sep_reversed
                ):
                    # i 是分隔符后的 token 数量，对应 allowed_tokens 中的 key
                    allowed_set = self.allowed_tokens.get(i, set())
                    # print(list(allowed_set))
                    return list(allowed_set)
            # 如果没有找到分隔符或者其他情况，可能需要返回一个默认允许的集合或所有 token
            # 这里根据原逻辑，如果没有匹配到，似乎没有返回值，这可能需要调整
            # 暂时返回空列表或一个默认值，例如 EOS token
            return list(
                self.allowed_tokens.get(0, {tokenizer.eos_token_id})
            )  # Fallback or default

        return prefix_allowed_tokens_fn

    def _process_data(self) -> None:
        """处理数据的抽象方法，子类必须实现。"""
        raise NotImplementedError


class SeqRecDataset(BaseDataset):
    """
    SeqRecDataset 类用于处理序列推荐数据集。

    该类继承自 BaseDataset，支持训练、验证和测试模式的数据处理。

    Attributes:
        mode (str): 数据处理模式，可以是 'train'、'valid' 或 'test'。
        prompt_sample_num (int): 每个样本的提示数量。
        prompt_id (int): 当前使用的提示 ID。
        sample_num (int): 样本数量。
        prompts (list): 提示列表。
        inter_data (list): 处理后的交互数据。
        valid_text_data (list): 验证模式下的文本数据。

    Methods:
        _load_data(): 加载数据集。
        _remap_items(): 重新映射数据项。
        _process_train_data(): 处理训练数据。
        _process_valid_data(): 处理验证数据。
        _process_test_data(): 处理测试数据。
        set_prompt(prompt_id): 设置提示 ID。
        __len__(): 返回数据集的长度。
        _construct_valid_text(): 构建验证文本数据。
        _get_text_data(data, prompt): 获取文本数据。
        __getitem__(index): 获取指定索引的数据项。
    """

    def __init__(
        self,
        args: Any,
        mode: str = "train",
        prompt_sample_num: int = 1,
        prompt_id: int = 0,
        sample_num: int = -1,
    ) -> None:
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["seqrec"]

        # load data
        self._load_data()
        self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self) -> None:
        """
        加载数据集。

        从指定路径加载交互数据和索引文件。
        """
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)

    def _remap_items(self) -> None:
        """
        重新映射数据项。

        将原始数据项重新映射为新的索引。
        """
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_train_data(self) -> List[Dict[str, Any]]:
        """
        处理训练数据。

        Returns:
            List[Dict[str, Any]]: 处理后的训练数据列表。
        """
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]
                if self.add_prefix:
                    history = [
                        str(k + 1) + ". " + item_idx
                        for k, item_idx in enumerate(history)
                    ]
                one_data["inters"] = self.his_sep.join(history)
                inter_data.append(one_data)

        return inter_data

    def _process_valid_data(self) -> List[Dict[str, Any]]:
        """
        处理验证数据。

        Returns:
            List[Dict[str, Any]]: 处理后的验证数据列表。
        """
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self) -> List[Dict[str, Any]]:
        """
        处理测试数据。

        Returns:
            List[Dict[str, Any]]: 处理后的测试数据列表。
        """
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id: int) -> None:
        """
        设置提示 ID。

        Args:
            prompt_id (int): 要设置的提示 ID。
        """
        self.prompt_id = prompt_id

    def __len__(self) -> int:
        """
        返回数据集的长度。

        Returns:
            int: 数据集的长度。
        """
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == "valid":
            return len(self.valid_text_data)
        elif self.mode == "test":
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self) -> None:
        """
        构建验证文本数据。
        """
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(
                    all_prompt_ids, self.prompt_sample_num, replace=False
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(
        self, data: Dict[str, Any], prompt: Dict[str, str]
    ) -> Tuple[str, str]:
        """
        获取文本数据。

        Args:
            data (Dict[str, Any]): 数据字典。
            prompt (Dict[str, str]): 提示字典。

        Returns:
            Tuple[str, str]: 输入和输出文本。
        """
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == "test":
            return input, response

        return input, output

    def __getitem__(self, index: int) -> Dict[str, str]:
        """
        获取指定索引的数据项。

        Args:
            index (int): 数据索引。

        Returns:
            Dict[str, str]: 包含输入和输出的字典。
        """
        if self.mode == "valid":
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]
        # print(index, idx)

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        # print({"input": input, "output": output})

        return dict(input_ids=input, labels=output)


class FusionSeqRecDataset(BaseDataset):
    def __init__(
        self, args, mode="train", prompt_sample_num=1, prompt_id=0, sample_num=-1
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["fusionseqrec"]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), "r") as f:
            self.item_feat = json.load(f)

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = "".join(self.indices[str(items[i])])
                one_data["title"] = (
                    self.item_feat[str(items[i])]["title"].strip().strip(".!?,;:`")
                )
                one_data["description"] = self.item_feat[str(items[i])]["description"]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]
                inters = ["".join(self.indices[str(j)]) for j in history]
                inter_titles = [
                    '"' + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + '"'
                    for j in history
                ]

                if self.add_prefix:
                    inters = [
                        str(k + 1) + ". " + item_idx
                        for k, item_idx in enumerate(inters)
                    ]
                    inter_titles = [
                        str(k + 1) + ". " + item_title
                        for k, item_title in enumerate(inter_titles)
                    ]

                one_data["inters"] = self.his_sep.join(inters)
                one_data["inter_titles"] = self.his_sep.join(inter_titles)
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-2])])
            one_data["title"] = (
                self.item_feat[str(items[-2])]["title"].strip().strip(".!?,;:`")
            )
            one_data["description"] = self.item_feat[str(items[-2])]["description"]

            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = [
                '"' + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + '"'
                for j in history
            ]

            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)
                ]
                inter_titles = [
                    str(k + 1) + ". " + item_title
                    for k, item_title in enumerate(inter_titles)
                ]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-1])])
            one_data["title"] = (
                self.item_feat[str(items[-1])]["title"].strip().strip(".!?,;:`")
            )
            one_data["description"] = self.item_feat[str(items[-1])]["description"]

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = [
                '"' + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + '"'
                for j in history
            ]

            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)
                ]
                inter_titles = [
                    str(k + 1) + ". " + item_title
                    for k, item_title in enumerate(inter_titles)
                ]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == "valid":
            return len(self.valid_text_data)
        elif self.mode == "test":
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(
                    all_prompt_ids, self.prompt_sample_num, replace=False
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == "test":
            return input, response

        return input, output

    def __getitem__(self, index):
        if self.mode == "valid":
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)


class ItemFeatDataset(BaseDataset):
    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        # load data
        self._load_data()
        self.feat_data = self._process_data()

    def _load_data(self):
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), "r") as f:
            self.item_feat = json.load(f)

    def _process_data(self):
        feat_data = []
        for iid in self.item_feat:
            feat = self.item_feat[iid]
            index = "".join(self.indices[iid])
            feat["item"] = index
            feat["title"] = feat["title"].strip().strip(".!?,;:`")
            feat_data.append(feat)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data

    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        return input, output

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)


class ItemSearchDataset(BaseDataset):
    def __init__(
        self, args, mode="train", prompt_sample_num=1, prompt_id=0, sample_num=-1
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["itemsearch"]

        # load data
        self._load_data()
        self.search_data = self._process_data()

    def _load_data(self):
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), "r") as f:
            self.user_info = json.load(f)

    def _process_data(self):
        search_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]
        user_vague_intention = self.user_info["user_vague_intention"]
        if self.mode == "train":
            user_vague_intention = user_vague_intention["train"]
        elif self.mode == "test":
            user_vague_intention = user_vague_intention["test"]
        else:
            raise NotImplementedError

        for uid in user_explicit_preference.keys():
            one_data = {}
            user_ep = user_explicit_preference[uid]
            user_vi = user_vague_intention[uid]["querys"]
            one_data["explicit_preferences"] = user_ep
            one_data["user_related_intention"] = user_vi[0]
            one_data["item_related_intention"] = user_vi[1]

            iid = user_vague_intention[uid]["item"]
            inters = user_vague_intention[uid]["inters"]

            index = "".join(self.indices[str(iid)])
            one_data["item"] = index

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len :]
            inters = ["".join(self.indices[str(i)]) for i in inters]
            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)
                ]

            one_data["inters"] = self.his_sep.join(inters)

            search_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.search_data) * self.prompt_sample_num
        elif self.mode == "test":
            return len(self.search_data)
        else:
            return len(self.search_data)

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == "test":
            return input, response

        return input, output

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num

        d = self.search_data[idx]
        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(
            random.choice(d["explicit_preferences"])
        )
        all_querys = [d["user_related_intention"], d["item_related_intention"]]
        d["query"] = random.choice(all_querys)

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)


class PreferenceObtainDataset(BaseDataset):
    def __init__(self, args, prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt["preferenceobtain"]

        # load data
        self._load_data()
        self._remap_items()

        self.preference_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), "r") as f:
            self.user_info = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):
        preference_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]

        for uid in user_explicit_preference.keys():
            one_data = {}
            inters = self.remapped_inters[uid][:-3]
            user_ep = user_explicit_preference[uid]

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len :]
            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)
                ]

            one_data["explicit_preferences"] = user_ep
            one_data["inters"] = self.his_sep.join(inters)

            preference_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.preference_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        return input, output

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num

        d = self.preference_data[idx]
        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(
            random.choice(d["explicit_preferences"])
        )

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)


class SeqRecTestDataset(BaseDataset):
    def __init__(self, args, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompt = all_prompt["seqrec"][self.prompt_id]

        # load data
        self._load_data()
        self._remap_items()

        self.inter_data = self._process_test_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_test_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)

            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

        self.prompt = all_prompt["seqrec"][self.prompt_id]

    def __len__(self):
        return len(self.inter_data)

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")

        return input, response

    def __getitem__(self, index):
        d = self.inter_data[index]
        input, target = self._get_text_data(d, self.prompt)

        return dict(input_ids=input, labels=target)
