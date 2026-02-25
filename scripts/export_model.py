from __future__ import annotations
import argparse
from pathlib import Path
from wood_defect.export import main as export_main

# This file exists just so you can run: python scripts/export_model.py ...
# It delegates to src/wood_defect/export.py

if __name__ == "__main__":
    export_main()
