"""Optional LLM commentary over the computed technical picture.

The model never sees raw candles and is never the source of numbers. It receives
the already-computed indicator readings and is asked to interpret them. If no API
key is configured, :func:`generate_commentary` returns ``None`` and the report is
still produced from deterministic analysis alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .config import LLMConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Anda adalah analis pasar XAUUSD. Anda menerima hasil teknikal yang sudah dihitung \
secara deterministik pada beberapa timeframe.

Aturan:
- Gunakan hanya data yang diberikan. Jangan mengarang harga, berita, atau angka.
- Jelaskan konteks sesi Tokyo, London, atau New York bila relevan.
- Jika timeframe bertentangan, katakan dengan jelas dan jangan memaksakan arah.
- Gunakan Bahasa Indonesia yang ringkas dan mudah dibaca di Telegram.
- AI hanya menjelaskan hasil mesin; jangan mengubah signal, skor, SL, atau TP.
- Jangan menyuruh pengguna membeli atau menjual. Jelaskan kondisi dan risiko.

Struktur jawaban:
1. Kondisi pasar (maksimal dua kalimat)
2. Keselarasan timeframe dan level penting
3. Hal yang membatalkan analisis
"""


class LLMError(RuntimeError):
    """The LLM call failed."""


def _post_chat_completion(config: LLMConfig, messages: list[dict]) -> str:
    import requests

    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=config.timeout)
    except Exception as exc:
        raise LLMError(f"request to {url} failed: {exc}") from exc

    if response.status_code >= 400:
        # Surface the provider's message; it usually names the real problem.
        raise LLMError(f"{response.status_code} from {url}: {response.text[:400]}")

    try:
        body = response.json()
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"unexpected response shape: {response.text[:400]}") from exc


def build_user_prompt(payload: Dict[str, Any]) -> str:
    """Render the analysis payload as the user message."""
    return (
        "Here is the computed technical data as JSON.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```\n\n"
        "Write the commentary as instructed."
    )


def generate_commentary(
    payload: Dict[str, Any],
    config: LLMConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return LLM commentary, or ``None`` when disabled or failing.

    Failure is non-fatal by design: losing optional prose must not discard a
    successfully computed report.
    """
    if not config.enabled:
        logger.info("LLM commentary skipped: no API key configured")
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(payload)},
    ]
    candidates = LLMConfig.candidates_from_env() or [config]
    attempted = []
    for candidate in candidates:
        attempted.append(candidate.provider)
        try:
            text = _post_chat_completion(candidate, messages).strip()
        except LLMError as exc:
            logger.warning("%s commentary unavailable: %s", candidate.provider, exc)
            continue
        if text:
            if metadata is not None:
                metadata.update(
                    {
                        "enabled": True,
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "attempted": attempted,
                    }
                )
            return text
    if metadata is not None:
        metadata.update({"enabled": False, "provider": None, "attempted": attempted})
    return None
