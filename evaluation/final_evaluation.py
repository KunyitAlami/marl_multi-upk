import os
import argparse

METHODS = ["fixed_time", "greedy_preemption", "mappo_nomask", "mappo_masked"]
JUMLAH_UNIT_LEVELS = [10, 15, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    from aggregate_results import aggregate
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hasil_evaluasi")
    if a.run:
        import eval_core
        for units in JUMLAH_UNIT_LEVELS:
            for m in METHODS:
                eval_core.run_batch(m, units)
    aggregate(root)


if __name__ == "__main__":
    main()
