"""Compatibility entry. Installed package required; history outside this project is untouched.

python scripts/09_render_exploratory.py --results-dir <saved run> --output-dir <new directory>
"""
import sys
from no2_ml.cli import main

if __name__=='__main__':
    main(['render',*sys.argv[1:]])
