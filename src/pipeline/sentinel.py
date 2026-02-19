"""Stage 3: Sentiment scorer using local Ollama or cloud LLM."""

import asyncio
import json
from datetime import datetime

import httpx
from loguru import logger

from ..config import get_settings
from ..models import SentimentResult, TechnicalSignal


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from response text (handles markdown code blocks)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


class SentimentScorer:
    """Scores sentiment using local Ollama or cloud LLM API."""

    def __init__(self, config=None):
        """Store settings and init HTTP client."""
        self.config = config or get_settings()
        self.use_local = self.config.USE_LOCAL_LLM
        self.client = httpx.AsyncClient(timeout=30.0)

    async def score(self, signal: TechnicalSignal) -> SentimentResult:
        """Score sentiment for a technical signal."""
        prompt = self._build_prompt(signal)
        if self.use_local:
            return await self._score_ollama(prompt, signal.symbol)
        return await self._score_cloud(prompt, signal.symbol)

    def _build_prompt(self, signal: TechnicalSignal) -> str:
        """Build the analysis prompt."""
        return f"""You are a crypto futures market analyst. Analyze the current market conditions for {signal.symbol}.

Technical context:
- Signal detected: {signal.signal_type.value}
- RSI (14): {signal.rsi:.1f}
- MACD Histogram: {signal.macd_histogram:.4f}
- ATR (14): {signal.atr:.4f}
- Funding Rate: {signal.funding_rate if signal.funding_rate is not None else 'N/A'}

Based on these technicals and your knowledge of crypto market dynamics, assess whether this is a genuine trading opportunity or likely a false signal.

Respond with ONLY valid JSON, no other text:
{{"score": <float between -1.0 and 1.0 where -1 is very bearish, 0 is neutral, 1 is very bullish>, "reason": "<one concise sentence explaining your assessment>"}}"""

    async def _score_ollama(self, prompt: str, symbol: str) -> SentimentResult:
        """Score using local Ollama API."""
        url = f"{self.config.OLLAMA_URL.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            raw = response.json().get("response", "")
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            score = max(-1.0, min(1.0, float(parsed.get("score", 0))))
            reason = str(parsed.get("reason", "No reason provided"))
            return SentimentResult(
                score=score,
                reason=reason,
                source="ollama",
                model_used=self.config.OLLAMA_MODEL,
                timestamp=datetime.utcnow(),
            )
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Ollama scoring failed for {}: {}", symbol, e)
            return SentimentResult(
                score=0.0,
                reason="Scoring failed — defaulting to neutral",
                source="ollama",
                model_used=self.config.OLLAMA_MODEL,
                timestamp=datetime.utcnow(),
            )

    async def _score_cloud(self, prompt: str, symbol: str) -> SentimentResult:
        """Call Anthropic or OpenAI API based on config.CLOUD_LLM_PROVIDER."""
        provider = (self.config.CLOUD_LLM_PROVIDER or "anthropic").lower()

        try:
            if provider == "anthropic":
                return await self._score_anthropic(prompt, symbol)
            if provider == "openai":
                return await self._score_openai(prompt, symbol)
            logger.warning("Unknown cloud provider {}, falling back to Anthropic", provider)
            return await self._score_anthropic(prompt, symbol)
        except Exception as e:
            logger.warning("Cloud scoring failed for {}: {}", symbol, e)
            return SentimentResult(
                score=0.0,
                reason="Cloud scoring failed — defaulting to neutral",
                source="cloud",
                model_used=self.config.CLOUD_LLM_MODEL,
                timestamp=datetime.utcnow(),
            )

    async def _score_anthropic(self, prompt: str, symbol: str) -> SentimentResult:
        """Call Anthropic Claude API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.config.CLOUD_LLM_API_KEY,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.config.CLOUD_LLM_MODEL,
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
                break
        parsed = _extract_json(text) or {}
        score = max(-1.0, min(1.0, float(parsed.get("score", 0))))
        reason = str(parsed.get("reason", "No reason provided"))
        return SentimentResult(
            score=score,
            reason=reason,
            source="cloud",
            model_used=self.config.CLOUD_LLM_MODEL,
            timestamp=datetime.utcnow(),
        )

    async def _score_openai(self, prompt: str, symbol: str) -> SentimentResult:
        """Call OpenAI Chat Completions API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.CLOUD_LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.CLOUD_LLM_MODEL,
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _extract_json(text) or {}
        score = max(-1.0, min(1.0, float(parsed.get("score", 0))))
        reason = str(parsed.get("reason", "No reason provided"))
        return SentimentResult(
            score=score,
            reason=reason,
            source="cloud",
            model_used=self.config.CLOUD_LLM_MODEL,
            timestamp=datetime.utcnow(),
        )

    async def score_batch(self, signals: list[TechnicalSignal]) -> list[SentimentResult]:
        """Score multiple signals concurrently."""
        tasks = [self.score(s) for s in signals]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
