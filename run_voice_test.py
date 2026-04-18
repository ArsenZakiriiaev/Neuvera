import json
import sys
from pathlib import Path

from neuvera_ai.service import run_voice_analysis_from_file


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: ./.venv/bin/python run_voice_test.py /path/to/audio.mp3")
        return 1

    audio_path = Path(sys.argv[1]).expanduser()
    if not audio_path.exists():
        print(json.dumps({"status": "failed", "error": f"File not found: {audio_path}"}, indent=2))
        return 1

    result = run_voice_analysis_from_file(audio_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
