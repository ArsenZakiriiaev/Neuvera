import json
import sys
from pathlib import Path

from neuvera_ai.service import run_tremor_analysis_from_video


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print("Usage: ./.venv/bin/python run_tremor_test.py /path/to/video.mp4 [--llm]")
        return 1

    video_path = Path(sys.argv[1]).expanduser()
    use_llm = len(sys.argv) == 3 and sys.argv[2] == "--llm"
    if len(sys.argv) == 3 and not use_llm:
        print("Only supported optional flag: --llm")
        return 1

    if not video_path.exists():
        print(json.dumps({"status": "failed", "error": f"File not found: {video_path}"}, indent=2))
        return 1

    result = run_tremor_analysis_from_video(video_path, use_llm=use_llm)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
