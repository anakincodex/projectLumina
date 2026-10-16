"""Subprocess execution runner that measures metrics across seeds in isolated processes."""
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, Optional


def inject_seed_and_reporter(code: str, seed: int) -> str:
    """
    Substitutes random seeds and injects a JSON metric collector at the end of the script.
    """
    # Replace common random_state seeds and np.random.seed calls
    seeded_code = re.sub(r"random_state\s*=\s*\d+", f"random_state={seed}", code)
    seeded_code = re.sub(r"np\.random\.seed\(\s*\d+\s*\)", f"np.random.seed({seed})", seeded_code)
    seeded_code = re.sub(r"random\.seed\(\s*\d+\s*\)", f"random.seed({seed})", seeded_code)

    collector = (
        "\n\n# --- [REPROVE METRIC COLLECTOR] ---\n"
        "import json\n"
        "_collected = {}\n"
        "for _var_name in ['test_auc', 'roc_auc', 'test_acc', 'accuracy', 'auc', 'acc', 'f1', 'test_f1']:\n"
        "    if _var_name in locals():\n"
        "        try:\n"
        "            _collected[_var_name] = float(locals()[_var_name])\n"
        "        except Exception:\n"
        "            pass\n"
        "print('__REPROVE_METRICS__:' + json.dumps(_collected))\n"
    )

    return seeded_code + collector


def execute_pipeline(
    code: str,
    seed: int,
    timeout: int = 15,
) -> Dict[str, float]:
    """
    Runs the pipeline code in a sandboxed subprocess for a specific random seed.
    Returns extracted metric dictionary.
    """
    instrumented_code = inject_seed_and_reporter(code, seed)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(instrumented_code)
        temp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Pipeline execution failed (exit {proc.returncode}):\n{proc.stderr}"
            )

        # Parse metrics from stdout
        for line in proc.stdout.splitlines():
            if line.startswith("__REPROVE_METRICS__:"):
                json_str = line.replace("__REPROVE_METRICS__:", "").strip()
                return json.loads(json_str)

        raise ValueError(
            f"No metrics captured from pipeline execution.\nStdout:\n{proc.stdout}\nStderr:\n{proc.stderr}"
        )

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
