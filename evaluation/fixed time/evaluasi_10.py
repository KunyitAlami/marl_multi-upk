import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_core

METHOD = "fixed_time"
JUMLAH_UNIT = 10

if __name__ == "__main__":
    eval_core.cli(METHOD, JUMLAH_UNIT)
