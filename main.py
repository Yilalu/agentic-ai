"""Run with: streamlit run main.py"""

from pathlib import Path
import runpy
import sys

# Put the project root on sys.path before importing config.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import PROJECT_ROOT

runpy.run_path(str(PROJECT_ROOT / "app.py"), run_name="__main__")
