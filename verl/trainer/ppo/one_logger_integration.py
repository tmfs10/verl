"""
onelogger_integration.py  – self‑contained
Define:

    class OneLoggerInstrumented    – mixin that auto‑patches its subclass
    inject_logging(trainer_cls)    – kept public for manual use elsewhere

Usage (simpler now)
-------------------
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

class LoggedRayPPOTrainer(OneLoggerInstrumented, RayPPOTrainer):
    pass

trainer = LoggedRayPPOTrainer(cfg, tok, role_map, pool_mgr, ...)
trainer.init_workers()
trainer.fit()
"""

from __future__ import annotations

import time
import os
from functools import wraps
from typing import Any, Callable, Tuple, List

from one_logger_utils import OneLoggerUtils

import ray

def _is_true_driver() -> bool:
    """
    Return True only for the one process that launched `ray.init()`
    (or Ray Client).  Works for tasks *and* actors.
    """
    try:
        ctx = ray.get_runtime_context()

        # In Ray driver these are *nil* (= all 0‑bytes).  In any worker they
        # have real IDs.
        from ray._raylet import ActorID, TaskID
        return (ctx.actor_id == ActorID.nil()) and (ctx.task_id == TaskID.nil())
    except Exception:
        # Ray not imported / not initialised → certainly the driver
        return True

class _NullLogger:
    def __getattr__(self, name):
        # every callback becomes a no‑op lambda
        return lambda *a, **kw: None

def get_rank():
    """Get the current process rank.
    
    In Ray environments, the trainer runs on the driver process and doesn't have
    the RANK environment variable set (which is set for Ray workers).
    Falls back to torch.distributed.get_rank() if available, otherwise defaults to 0.
    """
    try:
        # First try environment variable (set by Ray workers)
        return int(os.environ["RANK"])
    except KeyError:
        # If RANK not set, try torch.distributed if available
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank()
        except ImportError:
            pass
        # Default to rank 0 (driver process in Ray)
        return 0

# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #
def _now_ms() -> int:
    return int(time.time_ns() / 1_000_000)


# --------------------------------------------------------------------------- #
#  DataLoader proxy for batch‑level hooks                                     #
# --------------------------------------------------------------------------- #
class _LoggedDataLoader:
    def __init__(self, inner_loader, trainer, role: str):
        self._inner = inner_loader
        self._trainer = trainer
        self._role = role  # "train" or "val"

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __iter__(self):
        for batch in self._inner:
            if self._role == "train":
                self._trainer._ol("on_train_batch_start")
            else:
                self._trainer._ol("on_validation_batch_start")

            yield batch

            if self._role == "train":
                self._trainer._ol("on_train_batch_end")
            else:
                self._trainer._ol("on_validation_batch_end")

    def __len__(self):
        return len(self._inner)


def _wrap(
    trainer_cls,
    method_name: str,
    *,
    before: Callable | None = None,
    after: Callable | None = None,
    around: Callable | None = None,
):
    """
    Wrap `trainer_cls.method_name` unless that *function object* is already
    instrumented.  Works even when the method is merely inherited.
    """
    raw = getattr(trainer_cls, method_name)

    # Has somebody wrapped this exact function before?
    if getattr(raw, "_onelogger_wrapped", False):
        return

    if around is not None:
        @wraps(raw)
        def wrapped(self, *a, **kw):
            return around(self, raw, *a, **kw)
    else:
        @wraps(raw)
        def wrapped(self, *a, **kw):
            if before:
                before(self, *a, **kw)
            result = raw(self, *a, **kw)
            if after:
                after(self, result, *a, **kw)
            return result

    wrapped._onelogger_wrapped = True          # mark as done
    setattr(trainer_cls, method_name, wrapped)


# --------------------------------------------------------------------------- #
#  Core: inject all OneLogger hooks                                           #
# --------------------------------------------------------------------------- #
def inject_logging(trainer_cls):
    """Patch *trainer_cls* so its methods emit OneLogger callbacks."""
    # ---- dataloader creation ------------------------------------------------
    def _after_dl(self, *_):
        self.train_dataloader = _LoggedDataLoader(self.train_dataloader, self, "train")
        self.val_dataloader   = _LoggedDataLoader(self.val_dataloader,   self, "val")
        self._ol("on_dataloader_init_end", app_build_dataiters_finish_time=_now_ms())

    _wrap(
        trainer_cls,
        "_create_dataloader",
        before=lambda self, *a, **k: self._ol(
            "on_dataloader_init_start", app_build_dataiters_start_time=_now_ms()
        ),
        after=_after_dl,
    )

    # ---- model init ---------------------------------------------------------
    _wrap(
        trainer_cls,
        "init_workers",
        before=lambda self, *a, **k: self._ol("on_model_init_start"),
        after=lambda self, *_: self._ol("on_model_init_end"),
    )

    # ---- checkpoint loading -------------------------------------------------
    _wrap(
        trainer_cls,
        "_load_checkpoint",
        before=lambda self, *a, **k: self._ol("on_load_checkpoint_start"),
        after=lambda self, *_: self._ol("on_load_checkpoint_end"),
    )

    # ---- checkpoint saving (try/except) -------------------------------------
    def _around_save(self, fn, *a, **kw):
        gs = getattr(self, "global_steps", 0)
        self._ol("on_save_checkpoint_start", global_step=gs)
        try:
            out = fn(self, *a, **kw)
            self._ol("on_save_checkpoint_success", global_step=gs)
            return out
        finally:
            self._ol("on_save_checkpoint_end", global_step=gs)

    _wrap(trainer_cls, "_save_checkpoint", around=_around_save)

    # ---- validation ---------------------------------------------------------
    _wrap(
        trainer_cls,
        "_validate",
        before=lambda self, *a, **k: self._ol("on_validation_start"),
        after=lambda self, *_ret, **__: self._ol("on_validation_end"),
    )

    # ---- training loop (top) -----------------------------------------------
    raw_fit = trainer_cls.fit

    @wraps(raw_fit)
    def fit_with_logging(self, *a, **kw):
        self._ol(
            "on_train_start",
            train_iterations_start=getattr(self, "global_steps", 0),
            train_samples_start=0,
        )
        try:
            return raw_fit(self, *a, **kw)
        finally:
            self._ol("on_train_end")
            self._ol("on_app_end")

    trainer_cls.fit = fit_with_logging

    # ---- no‑op forwarders so external code can call them --------------------
    for cb_name in (
        "on_train_batch_start",
        "on_train_batch_end",
        "on_validation_batch_start",
        "on_validation_batch_end",
        "on_save_checkpoint_success",
    ):
        if not hasattr(trainer_cls, cb_name):
            setattr(
                trainer_cls,
                cb_name,
                lambda self, *a, _cb=cb_name, **kw: self._ol(_cb, *a, **kw),
            )

    trainer_cls._onelogger_patched = True  # sentinel
    return trainer_cls


# --------------------------------------------------------------------------- #
#  Mixin                                                                      #
# --------------------------------------------------------------------------- #
class OneLoggerInstrumented:
    """
    Inherit from this *before* the vendor trainer.

    It guarantees that the class is patched exactly once, then builds a
    OneLoggerUtils instance and provides the `_ol()` dispatcher.
    """

    # Auto‑patch when a new subclass is *defined*
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_onelogger_patched" not in cls.__dict__:
            inject_logging(cls)

    # --------------------------------------------------------------------- #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # upstream initialisation

        # Build OneLoggerUtils now that self.config exists
        world_sz = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        enable_rank0 = get_rank() == 0

        app_tag_run_version = "1.0.0"
        app_tag = f"{self.config.trainer.experiment_name}_{self.config.trainer.nnodes}_{self.config.data.train_batch_size}_{self.config.actor_rollout_ref.rollout.n}_{self.config.data.max_prompt_length}_{app_tag_run_version}"

        cfg = {
            "enable_for_current_rank": enable_rank0,
            "one_logger_async": True,
            "one_logger_project": "VeRL-Internal",
            "app_start_time": _now_ms(),
            "log_every_n_train_iterations": 50,
            "app_tag_run_version": app_tag_run_version,
            "summary_data_schema_version": "1.0.0",
            "app_run_type": "training",
            "app_tag": app_tag,
            "app_tag_run_name": self.config.trainer.experiment_name,
            "world_size": world_sz,
            "global_batch_size": self.config.data.train_batch_size,
            "batch_size": self.config.data.train_batch_size,
            "micro_batch_size": self.config.data.train_batch_size / world_sz,
            "train_iterations_target": self.total_training_steps,
            "train_samples_target": self.total_training_steps * self.config.data.train_batch_size,
            "is_train_iterations_enabled": True,
            "is_baseline_run": False,
            "is_test_iterations_enabled": False,
            "is_validation_iterations_enabled": True,
            "is_save_checkpoint_enabled": True,
            "is_log_throughput_enabled": False,
            "save_checkpoint_strategy": "sync",
        }
        self.one_logger_callbacks = OneLoggerUtils(cfg)
        self.one_logger_callbacks.on_dataloader_init_start(app_build_dataiters_start_time=self.app_build_dataiters_start_time)
        self.one_logger_callbacks.on_dataloader_init_end(app_build_dataiters_finish_time=self.app_build_dataiters_finish_time)

    # --------------------------------------------------------------------- #
    #  Convenience dispatcher                                                #
    # --------------------------------------------------------------------- #
    def _ol(self, fn_name: str, *a, **kw):
        cb = getattr(self, "one_logger_callbacks", None)
        if fn_name == 'on_dataloader_init_start':
            print(f"Calling _ol with fn_name: {fn_name}, a: {a}, kw: {kw}, cb: {cb}")
            self.app_build_dataiters_start_time = kw['app_build_dataiters_start_time']
        if fn_name == 'on_dataloader_init_end':
            print(f"Calling _ol with fn_name: {fn_name}, a: {a}, kw: {kw}, cb: {cb}")
            self.app_build_dataiters_finish_time = kw['app_build_dataiters_finish_time']
        if cb is not None:
            print(f"Calling _ol with fn_name: {fn_name}, a: {a}, kw: {kw}, cb: {cb}")
            getattr(cb, fn_name)(*a, **kw)
