"""Install CI dependencies, with bounded retries before any experiment starts."""

from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    # Keep the configured package index, TLS checks, version constraints and
    # already-satisfied packages. No dependency imports are needed to run this.
    command = [sys.executable, "-m", "pip", "--disable-pip-version-check", "install",
               "--timeout", "120", "--retries", "5",
               "-r", str(ROOT / "requirements.txt"), "tensorboard"]
    for attempt in range(1, 4):
        print(f"[DEPENDENCIES] Installation attempt {attempt}/3; socket timeout 120s.", flush=True)
        status = subprocess.run(command, cwd=ROOT).returncode
        if status == 0:
            print("[DEPENDENCIES] Installation succeeded; the experiment can start.", flush=True)
            return 0
        # A signal/interrupt must not turn cancellation into another attempt.
        if status < 0 or status in (130, 143):
            return status if status > 0 else 128 - status
        if attempt < 3:
            delay = 10 * attempt
            print(f"[DEPENDENCIES] pip exited with {status}; retrying in {delay}s. "
                  "Training/evaluation has not started.", flush=True)
            time.sleep(delay)
    print("[DEPENDENCIES] Installation failed after 3 attempts. Review the pip errors above; "
          "training/evaluation was not started.", file=sys.stderr, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
