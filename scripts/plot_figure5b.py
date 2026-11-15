"""Saved-prediction rendering only; supports --input-csv and --output-dir."""
import sys
from no2_ml.cli import main

if __name__=='__main__':
    main(['render',*sys.argv[1:]])
