"""Run all Lab 6 phases sequentially as isolated subprocesses."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path) -> None:
    cmd = [sys.executable, str(script_path)]
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path("lab6_encoder_decoder")
    run_script(root / "phase1_eng_hindi.py")
    run_script(root / "phase2_analysis.py")
    run_script(root / "phase3_eng_spanish.py")
    # Phase 4 (Summarization) removed — not required for this submission

    print("\n| Phase | Task                        | Epochs | Status        |")
    print("|-------|-----------------------------|--------|---------------|")
    print("| 1     | Eng→Hindi Translation       | 300    | COMPLETE      |")
    print("| 2     | Performance Analysis        | N/A    | COMPLETE      |")
    print("| 3     | Eng→Spanish (w/ Attention)  | 180    | COMPLETE      |")
    print("| 4     | Text Summarization          | N/A    | SKIPPED       |")
    print("=== ALL PHASES COMPLETE. LAB 6 DONE (Phase 4 skipped) ===")


if __name__ == "__main__":
    main()
