"""数据集工具模块，包含数据集加载和处理相关的函数。"""

from torch.utils.data import ConcatDataset

from data import (
    FusionSeqRecDataset,
    ItemFeatDataset,
    ItemSearchDataset,
    PreferenceObtainDataset,
    SeqRecDataset,
)


def load_datasets(args):
    """加载训练和验证数据集。

    Args:
        args: 包含数据集配置的参数对象。

    Returns:
        tuple: (训练数据集, 验证数据集)的元组。

    Raises:
        NotImplementedError: 当指定了未实现的任务类型时抛出。
    """
    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), (
        "prompt sample number does not match task number"
    )
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), (
        "data sample number does not match task number"
    )

    train_datasets = []
    for task, prompt_sample_num, data_sample_num in zip(
        tasks, train_prompt_sample_num, train_data_sample_num
    ):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "item2index" or task.lower() == "index2item":
            dataset = ItemFeatDataset(
                args,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "fusionseqrec":
            dataset = FusionSeqRecDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "itemsearch":
            dataset = ItemSearchDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "preferenceobtain":
            dataset = PreferenceObtainDataset(
                args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num
            )

        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args, "valid", args.valid_prompt_sample_num)

    return train_data, valid_data


def load_test_dataset(args):
    """加载测试数据集。

    Args:
        args: 包含数据集配置的参数对象。

    Returns:
        Dataset: 测试数据集。

    Raises:
        NotImplementedError: 当指定了未实现的任务类型时抛出。
    """
    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
        # test_data = SeqRecTestDataset(args, sample_num=args.sample_num)
    elif args.test_task.lower() == "itemsearch":
        test_data = ItemSearchDataset(args, mode="test", sample_num=args.sample_num)
    elif args.test_task.lower() == "fusionseqrec":
        test_data = FusionSeqRecDataset(args, mode="test", sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data