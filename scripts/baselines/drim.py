"""Launcher: runs src.baselines.drim when executed."""
import runpy
import os

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_root)
runpy.run_path("src/baselines/drim.py", run_name="__main__")
