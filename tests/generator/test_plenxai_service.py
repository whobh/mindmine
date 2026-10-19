import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from neurons.generator.services.plenxai_service import (
    CHECKPOINT_KIND_PLENXAI,
    PlenXAIService,
)
from neurons.generator.task_manager import GenerationTask, TaskStatus


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("response is not JSON")
        return self._payload


def make_task(parameters=None, checkpoint=None):
    return GenerationTask(
        task_id="task-123",
        modality="video",
        status=TaskStatus.PROCESSING,
        prompt="A red fox running through fresh snow",
        parameters=parameters or {},
        webhook_url="https://validator.example/callback",
        signed_by="validator-hotkey",
        created_at=1_700_000_000.0,
        checkpoint=checkpoint,
    )


def configure_service(monkeypatch):
    monkeypatch.setenv("PLENXAI_API_KEY", "pk_test_key")
    monkeypatch.setenv("PLENXAI_POLL_INTERVAL", "0")
    monkeypatch.setenv("PLENXAI_RETRY_DELAY", "0")
    return PlenXAIService()


def test_video_request_uses_omni_flash_and_returns_untouched_bytes(monkeypatch):
    service = configure_service(monkeypatch)
    original_media = b"\x00\x00\x00\x18ftypisom-provider-c2pa-video"
    submitted = []
    checkpoints = []
    statuses = iter(
        [
            {"success": True, "task_id": "px-42", "status": "queued"},
            {
                "success": True,
                "task_id": "px-42",
                "status": "completed",
                "result_url": "https://media.example/video.mp4",
                "error_message": None,
            },
        ]
    )

    def post(url, headers, json, timeout):
        submitted.append((url, headers, json, timeout))
        return FakeResponse(payload={"success": True, "task_id": "px-42"})

    def get(url, headers=None, timeout=None):
        if url.endswith("/status/px-42"):
            return FakeResponse(payload=next(statuses))
        assert url == "https://media.example/video.mp4"
        return FakeResponse(
            content=original_media,
            headers={"Content-Type": "video/mp4"},
        )

    monkeypatch.setattr(
        "neurons.generator.services.plenxai_service.requests.post", post
    )
    monkeypatch.setattr("neurons.generator.services.plenxai_service.requests.get", get)

    result = service.process_with_checkpoint(
        make_task(
            {
                "resolution": "480p",
                "duration": 5,
                "aspect_ratio": "9:16",
            }
        ),
        checkpoints.append,
    )

    assert result["data"] == original_media
    assert result["metadata"]["provider"] == "plenxai"
    assert result["metadata"]["model"] == "omni-flash-video"
    assert submitted == [
        (
            "https://plenxai.com/api/v1/developer/generate/video-omni",
            {
                "X-API-Key": "pk_test_key",
                "Content-Type": "application/json",
            },
            {
                "prompt": "A red fox running through fresh snow",
                "model": "omni-flash-video",
                "server": "flow",
                "duration": 4,
                "aspect_ratio": "9:16",
                "resolution": "720p",
            },
            60.0,
        )
    ]
    assert checkpoints == [
        {
            "kind": CHECKPOINT_KIND_PLENXAI,
            "task_id": "px-42",
            "modality": "video",
            "model": "omni-flash-video",
            "retries_used": 0,
        },
        None,
    ]


def test_resume_polls_existing_task_without_submitting_again(monkeypatch):
    service = configure_service(monkeypatch)

    def unexpected_post(*args, **kwargs):
        raise AssertionError("resume must not submit a second paid generation")

    def get(url, headers=None, timeout=None):
        if url.endswith("/status/existing-task"):
            return FakeResponse(
                payload={
                    "success": True,
                    "task_id": "existing-task",
                    "status": "completed",
                    "result_url": "https://media.example/existing.mp4",
                }
            )
        return FakeResponse(content=b"original-video")

    monkeypatch.setattr(
        "neurons.generator.services.plenxai_service.requests.post", unexpected_post
    )
    monkeypatch.setattr("neurons.generator.services.plenxai_service.requests.get", get)

    result = service.process(
        make_task(
            checkpoint={
                "kind": CHECKPOINT_KIND_PLENXAI,
                "task_id": "existing-task",
                "modality": "video",
                "model": "omni-flash-video",
                "retries_used": 0,
            }
        )
    )

    assert result["data"] == b"original-video"
    assert result["metadata"]["resumed"] is True


def test_http_200_queue_rejection_is_an_error(monkeypatch):
    service = configure_service(monkeypatch)
    monkeypatch.setattr(
        "neurons.generator.services.plenxai_service.requests.post",
        lambda *args, **kwargs: FakeResponse(
            payload={
                "success": False,
                "error": "concurrency_limit_exceeded",
                "message": "Wait for an active job to finish",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="concurrency_limit_exceeded"):
        service.process(make_task())


def test_only_video_is_supported(monkeypatch):
    service = configure_service(monkeypatch)

    assert service.supports_modality("video")
    assert not service.supports_modality("image")


def test_flow_server_is_always_selected(monkeypatch):
    service = configure_service(monkeypatch)

    assert service._build_payload(make_task())["server"] == "flow"


def test_registry_selects_plenxai_for_video(monkeypatch):
    monkeypatch.setitem(sys.modules, "c2pa", types.ModuleType("c2pa"))
    from neurons.generator.services.service_registry import ServiceRegistry

    monkeypatch.setenv("PLENXAI_API_KEY", "pk_test_key")
    monkeypatch.setenv("IMAGE_SERVICE", "none")
    monkeypatch.setenv("VIDEO_SERVICE", "plenxai")

    registry = ServiceRegistry()

    assert isinstance(registry.get_service("video"), PlenXAIService)
    assert registry.get_service("image") is None


@pytest.mark.skipif(
    os.getenv("RUN_PLENXAI_C2PA_TEST") != "1",
    reason="set RUN_PLENXAI_C2PA_TEST=1 to spend credits on a live generation",
)
def test_live_video_result_has_trusted_c2pa():
    if not os.getenv("PLENXAI_API_KEY"):
        pytest.fail("PLENXAI_API_KEY is required for the live C2PA test")

    verifier_path = (
        Path(__file__).parents[2] / "gas" / "verification" / "c2pa_verification.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plenxai_c2pa_verification", verifier_path
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    result = PlenXAIService().process(
        make_task(
            {
                "resolution": "720p",
                "duration": 4,
                "aspect_ratio": "16:9",
            }
        )
    )
    verification = verifier.verify_c2pa(result["data"])

    assert verification.verified, verification.error
    assert verification.signature_valid, verification.validation_errors
    assert verification.is_trusted_issuer, verification.cert_issuer
