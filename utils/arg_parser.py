"""参数解析模块，包含所有命令行参数的定义和处理。"""


def parse_global_args(parser):
    """解析全局参数。

    Args:
        parser: ArgumentParser对象。

    Returns:
        添加了参数的ArgumentParser对象。
    """
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--base_model", type=str, default="./llama-7b/", help="basic model path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./ckpt/", help="The output directory"
    )

    return parser


def parse_dataset_args(parser):
    """解析数据集相关参数。

    Args:
        parser: ArgumentParser对象。

    Returns:
        添加了参数的ArgumentParser对象。
    """
    parser.add_argument("--data_path", type=str, default="", help="data directory")
    parser.add_argument(
        "--tasks",
        type=str,
        default="seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain",
        help="Downstream tasks, separate by comma",
    )
    parser.add_argument("--dataset", type=str, default="Games", help="Dataset name")
    parser.add_argument(
        "--index_file", type=str, default=".index.json", help="the item indices file"
    )

    # arguments related to sequential task
    parser.add_argument(
        "--max_his_len",
        type=int,
        default=20,
        help="the max number of items in history sequence, -1 means no limit",
    )
    parser.add_argument(
        "--add_prefix",
        action="store_true",
        default=False,
        help="whether add sequential prefix in history",
    )
    parser.add_argument(
        "--his_sep", type=str, default=", ", help="The separator used for history"
    )
    parser.add_argument(
        "--only_train_response",
        action="store_true",
        default=False,
        help="whether only train on responses",
    )

    parser.add_argument(
        "--train_prompt_sample_num",
        type=str,
        default="1,1,1,1,1,1",
        help="the number of sampling prompts for each task",
    )
    parser.add_argument(
        "--train_data_sample_num",
        type=str,
        default="0,0,0,100000,0,0",
        help="the number of sampling prompts for each task",
    )

    parser.add_argument(
        "--valid_prompt_id", type=int, default=0, help="The prompt used for validation"
    )
    parser.add_argument(
        "--sample_valid",
        action="store_true",
        default=True,
        help="use sampled prompt for validation",
    )
    parser.add_argument(
        "--valid_prompt_sample_num",
        type=int,
        default=2,
        help="the number of sampling validation sequential recommendation prompts",
    )

    return parser


def parse_train_args(parser):
    """解析训练相关参数。

    Args:
        parser: ArgumentParser对象。

    Returns:
        添加了参数的ArgumentParser对象。
    """
    parser.add_argument(
        "--optim", type=str, default="adamw_torch", help="The name of the optimizer"
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        help="separate by comma",
    )
    parser.add_argument(
        "--lora_modules_to_save",
        type=str,
        default="embed_tokens,lm_head",
        help="separate by comma",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="either training checkpoint or final adapter",
    )

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")

    return parser


def parse_test_args(parser):
    """解析测试相关参数。

    Args:
        parser: ArgumentParser对象。

    Returns:
        添加了参数的ArgumentParser对象。
    """
    parser.add_argument("--ckpt_path", type=str, default="", help="The checkpoint path")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument(
        "--filter_items",
        action="store_true",
        default=False,
        help="whether filter illegal items",
    )

    parser.add_argument(
        "--results_file",
        type=str,
        default="./results/test-ddp.json",
        help="result output path",
    )

    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument(
        "--sample_num",
        type=int,
        default=-1,
        help="test sample number, -1 represents using all test data",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID when testing with single GPU"
    )
    parser.add_argument(
        "--test_prompt_ids",
        type=str,
        default="0",
        help="test prompt ids, separate by comma. 'all' represents using all",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
        help="test metrics, separate by comma",
    )
    parser.add_argument("--test_task", type=str, default="SeqRec")

    return parser
