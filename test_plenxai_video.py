#!/usr/bin/env python3
"""Generate exactly one PlenXAI Omni Flash video for installation testing."""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one 4-second 720p Omni Flash video through PlenXAI."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Environment file containing PLENXAI_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic close-up of a red fox walking through fresh snow, "
            "natural movement, realistic lighting"
        ),
        help="Prompt used for the single test generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_file).expanduser().resolve()
    if not env_path.is_file():
        print(f"TEST ERROR: Không tìm thấy file env: {env_path}", file=sys.stderr)
        return 1

    load_dotenv(env_path, override=False)
    if not os.getenv("PLENXAI_API_KEY", "").strip():
        print(
            f"TEST ERROR: Chưa có PLENXAI_API_KEY trong {env_path}",
            file=sys.stderr,
        )
        return 1

    # This installation test must never submit a replacement job automatically.
    os.environ["PLENXAI_MAX_RETRIES"] = "0"

    from neurons.generator.services.plenxai_service import PlenXAIService
    from neurons.generator.task_manager import GenerationTask, TaskStatus

    task = GenerationTask(
        task_id=f"plenxai-test-{int(time.time())}",
        modality="video",
        status=TaskStatus.PROCESSING,
        prompt=args.prompt,
        parameters={
            "resolution": "720p",
            "duration": 4,
            "aspect_ratio": "16:9",
        },
        webhook_url="",
        signed_by="local-install-test",
        created_at=time.time(),
    )

    print("Đang tạo MỘT video Omni Flash 4 giây, 720p qua PlenXAI...")
    try:
        result = PlenXAIService().process(task)
    except Exception as exc:
        print(f"TEST ERROR: {exc}", file=sys.stderr)
        return 1

    video = result.get("data")
    metadata = result.get("metadata") or {}
    if not isinstance(video, bytes) or not video:
        print("TEST ERROR: Provider không trả về dữ liệu video.", file=sys.stderr)
        return 1

    output_dir = Path(__file__).resolve().parent / "test-videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = str(metadata.get("task_id") or task.task_id)
    output_path = output_dir / f"{task_id}.mp4"
    output_path.write_bytes(video)

    print("TEST SUCCESS: PlenXAI đã gen và tải video thành công.")
    print(f"Model: {metadata.get('model', 'omni-flash-video')}")
    print(f"Task ID: {task_id}")
    print(f"Dung lượng: {len(video):,} bytes")
    print(f"Video: {output_path}")
    print("Lưu ý: Gen thành công chưa đồng nghĩa video đã pass C2PA của SN34.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
