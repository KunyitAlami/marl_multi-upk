
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import eval_core

METHOD = "greedy_preemption"
JUMLAH_UNIT_LIST = [10, 15, 20]


if __name__ == "__main__":
    for jumlah_unit in JUMLAH_UNIT_LIST:
        print("\n" + "=" * 60)
        print(f"Evaluasi {METHOD.upper()} - {jumlah_unit} Unit UPK")
        print("=" * 60)

        eval_core.cli(METHOD, jumlah_unit)
