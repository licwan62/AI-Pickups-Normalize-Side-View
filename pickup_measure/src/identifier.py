"""Identify a pickup from a single side-view image with Qwen vision.

The identifier returns structured make/model/trim/year fields.  The master
table is then searched with those fields; this module never guesses dimensions.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .settings import Settings


UNKNOWN_VALUES = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "unspecified",
    "未知",
}
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _clean_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ,;")
    return "" if text.casefold() in UNKNOWN_VALUES else text


def _parse_year(value: object) -> int | None:
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    years = [int(match) for match in YEAR_PATTERN.findall(str(value or ""))]
    return years[0] if years else None


@dataclass(frozen=True)
class VehicleIdentity:
    make: str
    model: str
    trim: str = ""
    year: int | None = None
    body_style: str = ""
    cab: str = ""
    bed: str = ""
    market: str = ""
    confidence: float = 0.0
    alternatives: tuple["VehicleIdentity", ...] = ()

    @property
    def query(self) -> str:
        parts = [self.make, self.model, self.trim, self.cab, self.bed]
        if self.year:
            parts.append(str(self.year))
        return " ".join(part for part in parts if part)

    def as_dict(self) -> dict[str, object]:
        return {
            "make": self.make,
            "model": self.model,
            "trim": self.trim,
            "year": self.year,
            "body_style": self.body_style,
            "cab": self.cab,
            "bed": self.bed,
            "market": self.market,
            "confidence": round(self.confidence, 4),
            "query": self.query,
            "alternatives": [item.as_dict() for item in self.alternatives],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "VehicleIdentity":
        if not isinstance(payload, dict):
            raise ValueError("Identifier response must be a JSON object")
        make = _clean_text(
            payload.get("make")
            or payload.get("brand")
            or payload.get("manufacturer")
        )
        model = _clean_text(payload.get("model"))
        if not make or not model:
            raise ValueError("Identifier response is missing make or model")
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        alternatives: list[VehicleIdentity] = []
        raw_alternatives = payload.get("alternatives")
        if isinstance(raw_alternatives, list):
            for raw in raw_alternatives[:3]:
                try:
                    alternatives.append(cls.from_payload(raw))
                except ValueError:
                    continue
        return cls(
            make=make,
            model=model,
            trim=_clean_text(payload.get("trim")),
            year=_parse_year(payload.get("year")),
            body_style=_clean_text(
                payload.get("body_style") or payload.get("body")
            ),
            cab=_clean_text(payload.get("cab") or payload.get("cab_style")),
            bed=_clean_text(payload.get("bed")),
            market=_clean_text(payload.get("market")).upper(),
            confidence=max(0.0, min(1.0, confidence)),
            alternatives=tuple(alternatives),
        )


class QwenVehicleIdentifier:
    """Ask Qwen to read make/model/trim/year from one side-view image."""

    def __init__(
        self,
        *,
        model: str = "qwen3-vl-plus",
        endpoint: str = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ),
        api_key: str = "",
        prompt_file: Path = Path("promting_identify.md"),
        timeout_seconds: float = 90.0,
        max_image_size: int = 1400,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.prompt_file = prompt_file
        self.timeout_seconds = timeout_seconds
        self.max_image_size = max_image_size
        self.last_response_content: str | None = None
        self.last_response_contents: list[str] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> "QwenVehicleIdentifier":
        detection = settings.detection
        return cls(
            model=detection.model,
            endpoint=detection.endpoint,
            api_key=detection.api_key,
            prompt_file=settings.master_table.identify_prompt_file,
            timeout_seconds=detection.timeout_seconds,
        )

    def _load_prompt(self) -> str:
        if not self.prompt_file.is_file():
            raise RuntimeError(
                f"Vehicle identification prompt not found: {self.prompt_file}"
            )
        prompt = self.prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise RuntimeError(
                f"Vehicle identification prompt is empty: {self.prompt_file}"
            )
        return prompt

    def _image_data_url(self, image: Image.Image) -> str:
        prepared = image.convert("RGB")
        if max(prepared.size) > self.max_image_size:
            prepared = prepared.copy()
            prepared.thumbnail(
                (self.max_image_size, self.max_image_size),
                Image.Resampling.LANCZOS,
            )
        stream = BytesIO()
        prepared.save(stream, format="JPEG", quality=90, optimize=True)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _extract_json(content: str) -> dict[str, object]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise RuntimeError("Qwen did not return a valid JSON object")

    def _request_content(
        self,
        image: Image.Image,
        prompt: str,
        validation_error: str | None = None,
    ) -> str:
        messages: list[dict[str, object]] = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image)},
                },
                {"type": "text", "text": prompt},
            ],
        }]
        if validation_error:
            messages.append({
                "role": "user",
                "content": (
                    "The previous answer failed validation: "
                    f"{validation_error}. Identify the vehicle in the image "
                    "again and return one complete JSON object only. "
                    "Do not use Markdown or explain."
                ),
            })
        request_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 900,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key.strip()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Qwen API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Qwen API connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Qwen API request timed out") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen API returned invalid response JSON") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen API response has no assistant content") from exc
        if not isinstance(content, str):
            raise RuntimeError("Qwen API assistant content is not text")
        return content

    def identify(self, image: Image.Image) -> VehicleIdentity:
        if not self.api_key.strip():
            raise RuntimeError(
                "Qwen API key is missing; set detection.api_key or "
                "DASHSCOPE_API_KEY"
            )
        self.last_response_content = None
        self.last_response_contents = []
        prompt = self._load_prompt()
        content = self._request_content(image, prompt)
        self.last_response_content = content
        self.last_response_contents.append(content)
        try:
            return VehicleIdentity.from_payload(self._extract_json(content))
        except (RuntimeError, ValueError) as exc:
            content = self._request_content(image, prompt, str(exc))
            self.last_response_content = content
            self.last_response_contents.append(content)
            return VehicleIdentity.from_payload(self._extract_json(content))
