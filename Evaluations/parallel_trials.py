"""Bounded process scheduling for independent held-out payment trials."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
from pathlib import Path


_WORKER = {}


def initialize_trial_worker(graph, cfg, model_path, report_dir):
    os.environ.update(
        RL_REPORT_FILE=str(Path(report_dir) / f"worker_{os.getpid()}.txt"),
        RL_FAST_VALIDATION="0", RL_FAST_ENV="0", CUDA_VISIBLE_DEVICES="",
    )
    import torch
    from Runtime import experiment_core as core

    torch.set_num_threads(1)
    core._install_ppo_loader_compatibility()
    model = core.PPO.load(str(model_path), device="cpu")
    _WORKER.update(core=core, graph=graph, cfg=cfg, model=model)
    print(f"[EVALUATION WORKER READY] pid={os.getpid()}; device=cpu", flush=True)


def run_trial_worker(task):
    from Runtime.paper_protocol import _run_trial

    transaction, index, training_seed, repetition, rate, rate_index = task
    result = _run_trial(
        _WORKER["core"], _WORKER["graph"], _WORKER["cfg"], _WORKER["model"],
        transaction, index, training_seed, repetition, rate, rate_index,
    )
    return index, result


def completed_trials(worker, tasks, workers, initializer=None, initargs=()):
    """Yield completed trials promptly, with at most 2 * workers in flight."""
    tasks = iter(tasks)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer, initargs=initargs,
    ) as executor:
        pending = set()

        def submit_next():
            try:
                task = next(tasks)
            except StopIteration:
                return
            pending.add(executor.submit(worker, task))

        try:
            for _ in range(workers * 2):
                submit_next()
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    yield future.result()
                    submit_next()
        finally:
            for future in pending:
                future.cancel()
