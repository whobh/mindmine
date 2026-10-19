import os
import time
from typing import Any, Dict, Optional

import bittensor as bt
import requests

from .base_service import BaseGenerationService, CheckpointFn
from ..task_manager import GenerationTask

API_BASE = "https://plenxai.com/api/v1/developer"
CHECKPOINT_KIND_PLENXAI = "plenxai_generation"
VIDEO_MODEL = "omni-flash-video"
VIDEO_SERVER = "flow"

VIDEO_RESOLUTION_MAP = {
    "480p": "720p",
    "720p": "720p",
    "1080p": "1080p",
    "4k": "4k",
}
VIDEO_ASPECT_RATIOS = {"16:9", "9:16"}
VIDEO_DURATIONS = (4, 6, 8, 10)
PENDING_STATUSES = {"queued", "processing"}


class _ProviderGenerationFailed(RuntimeError):
    pass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _nearest_duration(value: Any) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 4
    return min(VIDEO_DURATIONS, key=lambda duration: abs(duration - requested))


class PlenXAIService(BaseGenerationService):
    """Omni Flash text-to-video generation through PlenxAI's REST API."""

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv("PLENXAI_API_KEY", "").strip()
        self.base_url = os.getenv("PLENXAI_API_BASE_URL", API_BASE).rstrip("/")
        self.request_timeout = _env_float("PLENXAI_REQUEST_TIMEOUT", 60.0)
        self.poll_interval = _env_float("PLENXAI_POLL_INTERVAL", 4.0)
        self.poll_timeout = _env_float("PLENXAI_POLL_TIMEOUT", 1800.0)
        self.download_timeout = _env_float("PLENXAI_DOWNLOAD_TIMEOUT", 600.0)
        self.max_retries = min(3, max(0, int(_env_float("PLENXAI_MAX_RETRIES", 3.0))))
        self.retry_delay = _env_float("PLENXAI_RETRY_DELAY", 5.0)

        if self.api_key:
            bt.logging.info("PlenXAI Omni Flash service initialized")
        else:
            bt.logging.warning(
                "PLENXAI_API_KEY is not set; PlenXAI service is unavailable"
            )

    def is_available(self) -> bool:
        return bool(self.api_key)

    def supports_modality(self, modality: str) -> bool:
        return modality == "video"

    def get_supported_tasks(self) -> Dict[str, list]:
        return {"video": ["text_to_video"]}

    def get_api_key_requirements(self) -> Dict[str, str]:
        return {"PLENXAI_API_KEY": ("PlenXAI API key for Omni Flash video generation")}

    def process(self, task: GenerationTask) -> Dict[str, Any]:
        return self.process_with_checkpoint(task)

    def process_with_checkpoint(
        self, task: GenerationTask, on_checkpoint: CheckpointFn = None
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("PlenXAI service is unavailable: missing API key")
        if task.modality != "video":
            raise ValueError(f"PlenXAI supports video only, got {task.modality!r}")

        checkpoint_callback = on_checkpoint or (lambda _: None)
        parameters = task.parameters or {}
        poll_interval = float(
            parameters.get("plenxai_poll_interval", self.poll_interval)
        )
        poll_timeout = float(parameters.get("plenxai_poll_timeout", self.poll_timeout))
        started_at = time.monotonic()
        checkpoint = task.checkpoint
        resumed = checkpoint is not None
        retries_used = 0

        if checkpoint is not None:
            self._validate_checkpoint(checkpoint)
            try:
                retries_used = int(checkpoint.get("retries_used", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid PlenXAI checkpoint retries_used") from exc
            if not 0 <= retries_used <= self.max_retries:
                raise ValueError("Invalid PlenXAI checkpoint retries_used")
            if checkpoint.get("retry_pending"):
                checkpoint = None

        payload = self._build_payload(task)

        while True:
            if checkpoint is not None:
                task_id = checkpoint["task_id"]
                checkpoint = None
                bt.logging.info(f"Resuming PlenXAI video task {task_id}")
            else:
                task_id = self._submit(payload)
                checkpoint_callback(
                    {
                        "kind": CHECKPOINT_KIND_PLENXAI,
                        "task_id": task_id,
                        "modality": "video",
                        "model": VIDEO_MODEL,
                        "retries_used": retries_used,
                    }
                )

            try:
                status_result = self._poll(
                    task_id,
                    poll_interval=poll_interval,
                    poll_timeout=poll_timeout,
                )
                media_url = self._result_url(status_result)
                media, mime_type = self._download(media_url)
            except _ProviderGenerationFailed:
                if retries_used >= self.max_retries:
                    checkpoint_callback(None)
                    raise
                retries_used += 1
                checkpoint_callback(
                    {
                        "kind": CHECKPOINT_KIND_PLENXAI,
                        "modality": "video",
                        "model": VIDEO_MODEL,
                        "retry_pending": True,
                        "retries_used": retries_used,
                    }
                )
                bt.logging.warning(
                    "Retrying PlenXAI video generation "
                    f"({retries_used}/{self.max_retries})"
                )
                time.sleep(max(0.0, self.retry_delay))
                continue
            except Exception:
                checkpoint_callback(None)
                raise
            else:
                checkpoint_callback(None)
                break

        elapsed = time.monotonic() - started_at
        bt.logging.success(
            f"PlenXAI video task {task_id} downloaded "
            f"({len(media)} bytes in {elapsed:.1f}s)"
        )
        metadata = {
            "model": VIDEO_MODEL,
            "provider": "plenxai",
            "mime_type": mime_type,
            "task_id": task_id,
            "generation_time": elapsed,
        }
        if resumed:
            metadata["resumed"] = True
        return {"data": media, "metadata": metadata}

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, task: GenerationTask) -> Dict[str, Any]:
        parameters = task.parameters or {}
        raw_resolution = str(parameters.get("resolution", "720p")).lower()
        resolution = VIDEO_RESOLUTION_MAP.get(raw_resolution, "720p")
        aspect_ratio = parameters.get(
            "aspect_ratio", parameters.get("aspectRatio", "16:9")
        )
        if aspect_ratio not in VIDEO_ASPECT_RATIOS:
            aspect_ratio = "16:9"

        payload: Dict[str, Any] = {
            "prompt": task.prompt,
            "model": VIDEO_MODEL,
            "server": VIDEO_SERVER,
            "duration": _nearest_duration(parameters.get("duration", 4)),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
        return payload

    def _submit(self, payload: Dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/generate/video-omni",
            headers=self._headers(),
            json=payload,
            timeout=self.request_timeout,
        )
        result = self._json_response(response, "video submission")
        if result.get("success") is not True:
            error = result.get("error") or "submission_rejected"
            message = result.get("message") or "provider rejected the job"
            raise RuntimeError(f"PlenXAI video submission rejected: {error}: {message}")
        task_id = result.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise RuntimeError("PlenXAI video submission response omitted task_id")
        bt.logging.info(f"Submitted PlenXAI video task {task_id}")
        return task_id

    def _poll(
        self, task_id: str, poll_interval: float, poll_timeout: float
    ) -> Dict[str, Any]:
        deadline: Optional[float] = None
        if poll_timeout > 0:
            deadline = time.monotonic() + poll_timeout
        timeout_error = (
            f"PlenXAI video task {task_id} exceeded "
            f"{poll_timeout:g}s polling timeout"
        )

        while True:
            response = self._get_with_retries(
                f"{self.base_url}/status/{task_id}",
                operation="video status",
                headers=self._headers(),
                timeout=self.request_timeout,
                deadline=deadline,
                deadline_error=timeout_error,
            )
            self._raise_if_deadline_passed(deadline, timeout_error)
            result = self._json_response(response, "video status")
            if result.get("success") is not True:
                reason = (
                    result.get("error_message") or "provider returned success=false"
                )
                raise RuntimeError(f"PlenXAI video status failed: {reason}")
            status = str(result.get("status", "")).lower()

            if status == "completed":
                return result
            if status == "error":
                reason = result.get("error_message") or "provider reported failure"
                raise _ProviderGenerationFailed(
                    f"PlenXAI video task {task_id} failed: {reason}"
                )
            if status not in PENDING_STATUSES:
                raise RuntimeError(
                    f"PlenXAI video task {task_id} returned unknown status {status!r}"
                )

            sleep_for = max(0.0, poll_interval)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(timeout_error)
                sleep_for = min(sleep_for, remaining)
            time.sleep(sleep_for)

    @staticmethod
    def _result_url(result: Dict[str, Any]) -> str:
        url = result.get("result_url")
        if not isinstance(url, str) or not url.strip():
            raise RuntimeError("PlenXAI completed response omitted result_url")
        return url

    def _download(self, url: str) -> tuple[bytes, str]:
        response = self._get_with_retries(
            url,
            operation="video download",
            timeout=self.download_timeout,
        )
        media = response.content
        if not media:
            raise RuntimeError("PlenXAI video download returned empty media")
        mime_type = response.headers.get("Content-Type", "video/mp4").split(";", 1)[0]
        return media, mime_type

    def _get_with_retries(
        self,
        url: str,
        operation: str,
        timeout: float,
        headers: Optional[Dict[str, str]] = None,
        deadline: Optional[float] = None,
        deadline_error: Optional[str] = None,
    ) -> Any:
        for attempt in range(self.max_retries + 1):
            request_timeout = timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(deadline_error or "PlenXAI deadline exceeded")
                request_timeout = min(timeout, remaining)
            try:
                response = requests.get(url, headers=headers, timeout=request_timeout)
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
            else:
                if 200 <= response.status_code < 300:
                    return response
                if response.status_code < 500 and response.status_code != 429:
                    self._raise_for_status(response, operation)
                if attempt >= self.max_retries:
                    self._raise_for_status(response, operation)

            bt.logging.warning(
                f"Retrying PlenXAI {operation} request "
                f"({attempt + 1}/{self.max_retries})"
            )
            sleep_for = max(0.0, self.retry_delay)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(deadline_error or "PlenXAI deadline exceeded")
                sleep_for = min(sleep_for, remaining)
            time.sleep(sleep_for)

        raise RuntimeError(f"PlenXAI {operation} exhausted retries")

    @staticmethod
    def _raise_if_deadline_passed(
        deadline: Optional[float], error_message: str
    ) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(error_message)

    def _json_response(self, response: Any, operation: str) -> Dict[str, Any]:
        self._raise_for_status(response, operation)
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(f"PlenXAI {operation} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError(
                f"PlenXAI {operation} returned {type(result).__name__}, expected object"
            )
        return result

    @staticmethod
    def _raise_for_status(response: Any, operation: str) -> None:
        if 200 <= response.status_code < 300:
            return
        detail = response.text or "empty response"
        raise RuntimeError(
            f"PlenXAI {operation} failed: {response.status_code} - {detail}"
        )

    @staticmethod
    def _validate_checkpoint(checkpoint: Dict[str, Any]) -> None:
        if checkpoint.get("kind") != CHECKPOINT_KIND_PLENXAI:
            raise ValueError(
                f"Unsupported PlenXAI checkpoint kind {checkpoint.get('kind')!r}"
            )
        if not checkpoint.get("retry_pending") and not checkpoint.get("task_id"):
            raise ValueError("PlenXAI checkpoint omitted task_id")
        if checkpoint.get("modality") != "video":
            raise ValueError(
                "PlenXAI checkpoint modality does not match the restored task"
            )
