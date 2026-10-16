"""Isolated subprocess runner: executes a pipeline for one seed and captures its metrics.

The user's code is NOT rewritten to extract scores. Instead a small harness is prepended that:
  * wraps common sklearn.metrics functions so every computed score is recorded, and
  * (optionally) saves the arrays returned by the first train_test_split call, so the
    duplicate-row tool can hash the real train/test rows.
Metrics stored in well-known variable names (test_auc, test_acc, ...) take priority.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, Optional

METRIC_MARKER = "__REPROVE_METRICS__:"

# Variables we read at the end of the script (highest priority first in evaluator.METRIC_PRIORITY)
METRIC_VARIABLES = [
    "test_auc", "roc_auc", "auc", "test_acc", "accuracy", "acc", "test_accuracy",
    "test_f1", "f1", "test_r2", "r2", "test_score", "score",
    "test_mse", "mse", "test_rmse", "rmse", "test_mae", "mae",
]

# sklearn.metrics functions recorded by the harness (last call wins)
HOOKED_METRICS = [
    "roc_auc_score", "accuracy_score", "balanced_accuracy_score", "f1_score",
    "precision_score", "recall_score", "average_precision_score", "r2_score",
    "mean_squared_error", "mean_absolute_error", "log_loss",
]


class PipelineExecutionError(RuntimeError):
    """Raised when a pipeline cannot be executed or produces no metrics."""


HARNESS_PRELUDE = """# --- [REPROVE HARNESS] records sklearn metric calls and the first train/test split ---
import os as _rp_os
import json as _rp_json
_rp_metric_calls = {}
try:
    import sklearn.metrics as _rp_metrics
    def _rp_wrap(_rp_name, _rp_fn):
        def _rp_wrapped(*_a, **_k):
            _rp_value = _rp_fn(*_a, **_k)
            try:
                _rp_metric_calls[_rp_name] = float(_rp_value)
            except Exception:
                pass
            return _rp_value
        _rp_wrapped.__name__ = _rp_name
        _rp_wrapped.__doc__ = _rp_fn.__doc__
        return _rp_wrapped
    for _rp_name in %(hooked)r:
        if hasattr(_rp_metrics, _rp_name):
            setattr(_rp_metrics, _rp_name, _rp_wrap(_rp_name, getattr(_rp_metrics, _rp_name)))
except Exception:
    pass
_rp_split_dump = _rp_os.environ.get("REPROVE_SPLIT_DUMP")
if _rp_split_dump:
    try:
        import numpy as _rp_np
        import sklearn.model_selection as _rp_ms
        _rp_orig_split = _rp_ms.train_test_split
        _rp_split_saved = [False]
        def _rp_split(*_a, **_k):
            _rp_out = _rp_orig_split(*_a, **_k)
            if not _rp_split_saved[0] and len(_rp_out) >= 2:
                try:
                    _rp_np.savez(_rp_split_dump, X_train=_rp_np.asarray(_rp_out[0]), X_test=_rp_np.asarray(_rp_out[1]))
                    _rp_split_saved[0] = True
                except Exception:
                    pass
            return _rp_out
        _rp_ms.train_test_split = _rp_split
    except Exception:
        pass
try:
    display
except NameError:
    def display(*_a, **_k):
        print(*_a)
# --- [END REPROVE HARNESS] ---
""" % {"hooked": HOOKED_METRICS}


METRIC_COLLECTOR = """

# --- [REPROVE METRIC COLLECTOR] ---
_rp_collected = {}
for _rp_var in %(variables)r:
    if _rp_var in globals():
        try:
            _rp_collected[_rp_var] = float(globals()[_rp_var])
        except Exception:
            pass
for _rp_key, _rp_val in _rp_metric_calls.items():
    _rp_collected.setdefault(_rp_key, _rp_val)
print(%(marker)r + _rp_json.dumps(_rp_collected))
""" % {"variables": METRIC_VARIABLES, "marker": METRIC_MARKER}


def inject_seed_and_reporter(code: str, seed: int) -> str:
    """Replace hard-coded seeds with `seed`, prepend the harness and append the metric collector."""
    seeded = re.sub(r"random_state\s*=\s*\d+", f"random_state={seed}", code)
    seeded = re.sub(r"np\.random\.seed\(\s*\d+\s*\)", f"np.random.seed({seed})", seeded)
    seeded = re.sub(r"(?<![\w.])random\.seed\(\s*\d+\s*\)", f"random.seed({seed})", seeded)
    return HARNESS_PRELUDE + "\n" + seeded + METRIC_COLLECTOR


def _short_error(stderr: str, max_lines: int = 6) -> str:
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    return "\n".join(lines[-max_lines:]) if lines else "(no error output)"


def execute_pipeline(
    code: str,
    seed: int,
    timeout: int = 60,
    cwd: Optional[str] = None,
    split_dump_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run the pipeline in a separate Python process (with a timeout) for one seed.
    Returns the captured metric dictionary. Raises PipelineExecutionError on failure.

    Note: this is process isolation with a timeout, not a security sandbox. Only audit code you trust.
    """
    instrumented = inject_seed_and_reporter(code, seed)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(instrumented)
        temp_path = f.name

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"            # never open plot windows
    env["PYTHONIOENCODING"] = "utf-8"     # avoid Windows console encoding crashes
    env["PYTHONHASHSEED"] = str(seed)
    env.pop("REPROVE_SPLIT_DUMP", None)
    if split_dump_path:
        env["REPROVE_SPLIT_DUMP"] = split_dump_path

    try:
        try:
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=cwd or None,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise PipelineExecutionError(f"Timed out after {timeout}s (seed {seed}).")

        if proc.returncode != 0:
            raise PipelineExecutionError(
                f"Exited with code {proc.returncode} (seed {seed}):\n{_short_error(proc.stderr)}"
            )

        for line in proc.stdout.splitlines():
            if line.startswith(METRIC_MARKER):
                return json.loads(line[len(METRIC_MARKER):].strip())

        raise PipelineExecutionError("The script finished but printed no metric marker.")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
