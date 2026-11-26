"""Together AI 模型配置与消息调用。模型名称集中在这里，便于公平地复现实验。"""

import logging
import os
from functools import lru_cache

import openai


DEFAULT_CANDIDATES = (
    "Qwen/Qwen3.5-9B",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "MiniMaxAI/MiniMax-M3.1-Flash",
)


def get_model(role: str) -> str:
    # 模型的「角色」指它在本轮做的事。同一个模型既能回答，也能做路由；
    # 给 judge 单独设定模型后，各种方法就可以由同一位裁判评分。
    return (os.getenv(f"CS329_{role.upper()}_MODEL") or os.getenv("CS329_MODEL")
            or DEFAULT_CANDIDATES[0])


def get_candidate_models() -> list[str]:
    configured = os.getenv("CS329_CANDIDATE_MODELS")
    if not configured:
        return list(DEFAULT_CANDIDATES)
    models = [name.strip() for name in configured.split(",")]
    if len(models) != 3 or len(set(models)) != 3 or not all(models):
        raise ValueError("CS329_CANDIDATE_MODELS 需要三个不同的模型名，用英文逗号分隔。")
    return models


@lru_cache(maxsize=1)
def get_together_client():
    key = os.getenv("TOGETHER_API_KEY")
    if not key:
        raise ValueError("模型调用需要 TOGETHER_API_KEY，请在仓库的 .env 中设置，再运行配置单元。")
    # SDK 会对限流、连接中断和可重试的服务端错误进行退避重试。
    return openai.OpenAI(api_key=key, base_url="https://api.together.xyz/v1",
                         timeout=60.0, max_retries=2)


def generate_together(model, messages, temperature=0.7, response_format=None, max_tokens=None):
    """返回聊天消息；服务请求失败时返回 None，调用者可据此补查或停止评估。"""
    client = get_together_client()
    args = {"model": model, "messages": messages, "temperature": temperature}
    if response_format is not None:
        args["response_format"] = response_format
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if model == "Qwen/Qwen3.5-9B" and (response_format is not None or max_tokens is not None):
        # 工具参数和研究计划需要完整的 JSON。给短输出留足生成空间，
        # 避免该模型的内部推理先用完这一轮的 token 预算。
        args["extra_body"] = {"reasoning": {"enabled": False}}
    try:
        response = client.chat.completions.create(**args)
        return response.choices[0].message if response.choices else None
    except openai.APIError as exc:
        logging.getLogger(__name__).warning(
            "模型 %s 调用失败 (%s, HTTP %s)", model, type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        return None
