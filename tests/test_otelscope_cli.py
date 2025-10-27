import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "tests" / "data" / "sample_otelcol_debug_output.txt"


def run_cli(args):
    result = subprocess.run(
        [sys.executable, "otelscope.py", *args],
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        check=True,
    )
    return result.stdout


def test_default_includes_all_types():
    output = run_cli(["--otelcol_output", str(SAMPLE_PATH)])

    assert "Trace ID:" in output
    assert "Span ID:" in output
    assert "Metric:" in output
    assert "Log:" in output
    # Operation name annotations should appear
    assert "[op:chat]" in output


def test_telemetry_span_filters_logs_and_metrics():
    output = run_cli(["--otelcol_output", str(SAMPLE_PATH), "--telemetry", "span"])
    assert "Trace ID:" in output
    assert "Span ID:" in output
    assert "Metric:" not in output
    assert "Log:" not in output
    assert "[op:chat]" in output  # spans still annotated
