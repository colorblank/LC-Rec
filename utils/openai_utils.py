"""OpenAI API 工具模块，包含与 OpenAI API 交互相关的功能。"""

import time
import openai


def get_res_batch(
    model_name: str, prompt_list: list[str], max_tokens: int, api_info: dict
) -> list[str]:
    """使用OpenAI API批量获取文本补全结果。

    Args:
        model_name: OpenAI模型名称。
        prompt_list: 提示文本列表。
        max_tokens: 生成文本的最大token数。
        api_info: 包含API密钥列表的字典。

    Returns:
        包含生成文本的列表。如果发生错误则返回None。

    Raises:
        各种OpenAI API异常,包括认证错误、速率限制等。
    """
    while True:
        try:
            res = openai.Completion.create(
                model=model_name,
                prompt=prompt_list,
                temperature=0.4,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            output_list = []
            for choice in res["choices"]:
                output = choice["text"].strip()
                output_list.append(output)

            return output_list

        except openai.error.AuthenticationError as e:
            print(e)
            openai.api_key = api_info["api_key_list"].pop()
            time.sleep(10)
        except openai.error.RateLimitError as e:
            print(e)
            if (
                str(e)
                == "You exceeded your current quota, please check your plan and billing details."
            ):
                openai.api_key = api_info["api_key_list"].pop()
                time.sleep(10)
            else:
                print("\nopenai.error.RateLimitError\nRetrying...")
                time.sleep(10)
        except openai.error.ServiceUnavailableError as e:
            print(e)
            print("\nopenai.error.ServiceUnavailableError\nRetrying...")
            time.sleep(10)
        except openai.error.Timeout:
            print("\nopenai.error.Timeout\nRetrying...")
            time.sleep(10)
        except openai.error.APIError as e:
            print(e)
            print("\nopenai.error.APIError\nRetrying...")
            time.sleep(10)
        except openai.error.APIConnectionError as e:
            print(e)
            print("\nopenai.error.APIConnectionError\nRetrying...")
            time.sleep(10)
        except Exception as e:
            print(e)
            return None
