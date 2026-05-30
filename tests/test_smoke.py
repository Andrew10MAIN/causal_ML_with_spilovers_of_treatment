import subprocess

def test_imports():
    import src.data.simulation
    import src.models.modelling


def test_pipeline_runs():
    result = subprocess.run(
        ["python", "pipelines/run_all.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0

from src.utils.config import load_config

def test_config():
    cfg = load_config()
    assert "simulation" in cfg