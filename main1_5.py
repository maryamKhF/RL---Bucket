"""main1/5: train round one, save artifacts, then exit before testing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args(argv=None, *, phase="train"):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=(__doc__ if phase == "train" else
        "Test round one from completed training artifacts; never start training."))
    parser.add_argument("--snapshot", type=Path, default=root / "20230716.gml.geo",
                        help="Network snapshot for a fresh run (default: 20230716.gml.geo).")
    parser.add_argument("--config", type=Path, default=root / "Configs/config.yaml")
    parser.add_argument("--output-dir", type=Path, default=root / "theResult")
    parser.add_argument("--failure-rates", type=float, nargs="+",
                        help="Default: all six rates in the configuration.")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-run", type=Path, help="Continue an extracted run directory.")
    resume.add_argument("--resume-archive", type=Path, help="Restore and continue a recovery or GitLab artifact ZIP.")
    resume.add_argument("--resume-job", help="Download and continue artifacts from this GitLab job ID.")
    parser.add_argument("--backup-seconds", type=float, default=300,
                        help="Legacy compatibility option; ZIPs are now built only at phase exit.")
    parser.add_argument("--checkpoint-seconds", type=float, default=1800,
                        help="Minimum interval for local coherent training checkpoints (default: 1800 seconds). "
                             "Initial/final checkpoints are always saved; 0 saves at every PPO update.")
    parser.add_argument("--package-only", action="store_true",
                        help="Package existing runs without training/testing (CI after_script).")
    args = parser.parse_args(argv)
    if phase == "evaluate" and not args.package_only and not any(
            (args.resume_run, args.resume_archive, args.resume_job)):
        parser.error("Testing requires --resume-run, --resume-archive or --resume-job from completed training.")
    return args


def main(argv=None, *, phase="train"):
    args = parse_args(argv, phase=phase)
    if args.package_only:
        from Runtime.round_one import package_round
        return [package_round(path) for path in args.output_dir.glob("main1_5_*")
                if path.is_dir() and (path / "run_metadata.json").is_file()]
    if args.resume_job:
        from Runtime.round_one_recovery import download_job_artifacts
        args.resume_archive = download_job_artifacts(args.resume_job, args.output_dir)
    # Set limits before importing numpy/torch, including in spawned workers.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    from Runtime.round_one import MainOneOfFive
    return MainOneOfFive(args.snapshot, args.config, args.output_dir,
                         failure_rates=args.failure_rates, resume_run=args.resume_run,
                         resume_archive=args.resume_archive, backup_seconds=args.backup_seconds,
                         checkpoint_seconds=args.checkpoint_seconds).run(phase=phase)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
