"""文本处理工具模块，包含文本清理和处理相关的功能。"""

import html
import re


def clean_text(raw_text: str | list | dict) -> str:
    """清理文本,移除HTML标签和特殊字符。

    Args:
        raw_text: 需要清理的原始文本,可以是字符串、列表或字典。

    Returns:
        清理后的文本字符串。如果清理后文本长度超过2000,则返回空字符串。
    """
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r"</?>\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            new_raw_text.append(raw.strip())
        cleaned_text = " ".join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?>\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text