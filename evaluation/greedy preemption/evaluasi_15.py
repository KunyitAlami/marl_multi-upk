import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_core

METHOD = "greedy_preemption"
JUMLAH_UNIT = 15

if __name__ == "__main__":
    eval_core.cli(METHOD, JUMLAH_UNIT)
