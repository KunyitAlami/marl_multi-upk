import os
import glob
import pandas as pd

METHODS = ["fixed_time", "greedy_preemption", "mappo_nomask", "mappo_masked"]
# Semua metrik: nilai lebih kecil = lebih baik.
# tt_upk_cens / wt_upk_cens = metrik UTAMA (UPK yang belum sampai dihitung
# sampai akhir simulasi). tt_upk / wt_upk = hanya UPK yang sampai.
METRICS = ["tt_upk_cens", "wt_upk_cens", "tt_upk", "wt_upk", "tt_non_upk", "ql_non_upk", "wt_non_upk"]
# Metrik pendukung (lebih besar = lebih baik), hanya masuk ringkasan.
EXTRA = ["upk_arrival_rate", "upk_arrived", "upk_spawned", "teleports_total", "teleports_upk"]
PROPOSED = "mappo_masked"


def aggregate(result_root):
    files = [f for m in METHODS for f in sorted(glob.glob(os.path.join(result_root, m, "runs_*.csv")))]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Membaca {len(files)} file hasil, {len(df)} simulasi")
    print(df.groupby("method").size().to_string())

    keys = ["demand_level", "jumlah_unit"]
    summ = df.groupby(keys + ["method"])[METRICS + EXTRA].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(os.path.join(result_root, "ringkasan_mean_std.csv"), index=False, encoding="utf-8-sig")

    rows = []
    for (d, u), sub in df.groupby(keys):
        for m in METRICS:
            means = sub.groupby("method")[m].mean()
            piv = sub.pivot(index="seed", columns="method", values=m)
            row = {"demand_level": d, "jumlah_unit": u, "metric": m, **{f"mean_{k}": v for k, v in means.items()}, "best_method": means.idxmin()}
            if PROPOSED in means:
                row["proposed_rank"] = int(means.rank(method="min")[PROPOSED])
                for k in means.index:
                    if k == PROPOSED:
                        continue
                    row[f"improve_pct_vs_{k}"] = 100 * (means[k] - means[PROPOSED]) / means[k] if means[k] else float("nan")
                    row[f"seeds_better_vs_{k}"] = int((piv[PROPOSED] < piv[k]).sum()) if k in piv else None
            rows.append(row)
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(result_root, "perbandingan_metode.csv"), index=False, encoding="utf-8-sig")

    n_scn = len(df.groupby(keys))
    print(f"\n=== Jumlah skenario (dari {n_scn}) di mana tiap metode terbaik, per metrik ===")
    print(cmp.groupby(["metric", "best_method"]).size().unstack(fill_value=0).reindex(METRICS).to_string())
    print(f"\nTotal (skenario x metrik) terbaik: {cmp['best_method'].value_counts().to_dict()} dari {len(cmp)}")

    print("\n=== Rata-rata upk_arrival_rate (UPK sampai / UPK di-spawn), per demand ===")
    print(df.pivot_table(index="method", columns="demand_level", values="upk_arrival_rate", aggfunc="mean").reindex(METHODS).round(3).to_string())

    print(f"\nFile: {os.path.join(result_root, 'ringkasan_mean_std.csv')}\n      {os.path.join(result_root, 'perbandingan_metode.csv')}")
    return summ, cmp


if __name__ == "__main__":
    import sys
    aggregate(sys.argv[1])