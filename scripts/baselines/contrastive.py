"""Launcher: runs src.baselines.contrastive when executed."""
import runpy
import os

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_root)
runpy.run_path("src/baselines/contrastive.py", run_name="__main__")
