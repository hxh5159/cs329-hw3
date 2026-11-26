"""按实验角色选择模型，支持 Together 和其他 OpenAI 兼容接口。"""

import logging
import os
from functools import lru_cache
from urllib.parse import urlsplit

import httpx
import openai


logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = (
    "Qwen/Qwen3.5-9B",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "MiniMaxAI/MiniMax-M3.1-Flash",
)

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def get_base_url() -> str:
    url = (os.getenv("CS329_BASE_URL") or TOGETHER_BASE_URL).rstrip("/")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("CS329_BASE_URL 需要填写服务商提供的 HTTPS API 地址。")
    return url


def get_structured_output_mode() -> str:
    default = "json_schema" if get_base_url() == TOGETHER_BASE_URL else "prompt"
    mode = os.getenv("CS329_STRUCTURED_OUTPUT") or default
    if mode not in {"json_schema", "json_object", "prompt"}:
        raise ValueError("CS329_STRUCTURED_OUTPUT 支持 json_schema、json_object 或 prompt。")
    return mode


def get_model(role: str) -> str:
    # 模型的「角色」指它在本轮做的事。同一个模型既能回答，也能做路由；
    # 给 judge 单独设定模型后，各种方法就可以由同一位裁判评分。
    model = os.getenv(f"CS329_{role.upper()}_MODEL") or os.getenv("CS329_MODEL")
    if model:
        return model
    if get_base_url() != TOGETHER_BASE_URL:
        raise ValueError("自定义接口需要在 CS329_MODEL 中指定该账户可以调用的模型。")
    return DEFAULT_CANDIDATES[0]


def get_candidate_models() -> list[str]:
    configured = os.getenv("CS329_CANDIDATE_MODELS")
    if not configured:
        if get_base_url() != TOGETHER_BASE_URL:
            raise ValueError("请为当前服务配置 CS329_CANDIDATE_MODELS 的三个融合候选。")
        return list(DEFAULT_CANDIDATES)
    models = [name.strip() for name in configured.split(",")]
    if len(models) != 3 or len(set(models)) != 3 or not all(models):
        raise ValueError("CS329_CANDIDATE_MODELS 需要三个不同的模型名，用英文逗号分隔。")
    return models


def get_together_client():
    url = get_base_url()
    return _client_for(url)


@lru_cache(maxsize=1)
def _client_for(url):
    # 单独配置中转站密钥，防止切换地址时把旧服务商的密钥带到新站点。
    key_name = "TOGETHER_API_KEY" if url == TOGETHER_BASE_URL else "CS329_API_KEY"
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"模型调用需要 {key_name}，请配置后重启内核。")
    # 长报告需要较长的读取等待时间，建立连接仍有单独的短时限。
    # 这里调整客户端的等待时间，服务商的生成设置和 SDK 的重试次数沿用原配置。
    return openai.OpenAI(
        api_key=key,
        base_url=url,
        timeout=httpx.Timeout(240.0, connect=15.0),
        max_retries=2,
        http_client=openai.DefaultHttpxClient(follow_redirects=False),
    )


def generate_together(model, messages, temperature=0.7, response_format=None, max_tokens=None):
    """返回聊天消息，服务请求失败时返回 None，并记录经过筛选的失败状态。"""
    client = get_together_client()
    args = {"model": model, "messages": messages, "temperature": temperature}
    if response_format is not None:
        args["response_format"] = response_format
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if get_base_url() == TOGETHER_BASE_URL and model == "Qwen/Qwen3.5-9B" and (response_format is not None or max_tokens is not None):
        # 工具参数和研究计划需要完整的 JSON。给短输出留足生成空间，
        # 避免该模型的内部推理先用完这一轮的 token 预算。
        args["extra_body"] = {"reasoning": {"enabled": False}}
    try:
        response = client.chat.completions.create(**args)
    except openai.APIError as exc:
        status = getattr(exc, "status_code", None)
        http_status = f"HTTP {status}" if isinstance(status, int) else "无 HTTP 状态"
        logger.warning("模型调用失败 (%s, %s)", type(exc).__name__, http_status)
        return None

    choices = getattr(response, "choices", None)
    if not choices:
        logger.warning("模型接口未返回聊天候选 (finish_reason=no_choice)。")
        return None
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        reason = getattr(choice, "finish_reason", None)
        if reason not in ("stop", "length", "tool_calls", "content_filter", "function_call"):
            reason = "unknown"
        # 不记录响应内容或推理文字，也不打印服务商提供的原始错误。
        logger.warning("模型接口未返回回答文本 (finish_reason=%s)。", reason)
    return message
