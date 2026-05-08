import sys
from pathlib import Path

# Make app and training modules importable from anywhere pytest is invoked
sys.path.insert(0, str(Path(__file__).parent / "app"))
sys.path.insert(0, str(Path(__file__).parent / "training"))
