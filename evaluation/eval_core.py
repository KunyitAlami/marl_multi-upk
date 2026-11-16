import os
import sys
import csv
import time
import traceback
import argparse
import importlib
import io
import contextlib
from datetime import datetime
import numpy as np
import torch

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
for p in (PROJECT_ROOT, EVAL_DIR, os.path.join(EVAL_DIR, "greedy preemption"), os.path.join(PROJECT_ROOT, "training"), os.path.join(PROJECT_ROOT, "environment")):
    if p not in sys.path:
        sys.path.insert(0, p)

import libsumo as traci
from environment.env import TrafficEnvironment, MIN_GREEN_DURATION
from environment.state import state
from environment.action import action
import greedy_premption as gp
from metric_evaluation import arrivalTimeUPK, waitingTimeUPK, travelTimeNonEMV, queueLengthNonEMV, waitingTimeNonEMV

_orig_traci_start = traci.start


def _quiet_start(cmd, *args, **kwargs):
    return _orig_traci_start(list(cmd) + ["--no-warnings"], *args, **kwargs)


traci.start = _quiet_start
MAX_SIMULATION_TIME = 3600
MAX_EXTENDED_TIME = 86400
UPK_MISS_LIMIT = 60

RESULT_ROOT = os.path.join(PROJECT_ROOT, "hasil_evaluasi")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_MASKED = os.path.join(PROJECT_ROOT, "hasil_train_model", "mappo", "mappo_20261021_123520", "mappo_final.pt")
MODEL_NOMASK = os.path.join(PROJECT_ROOT, "hasil_train_model_nomask", "mappo_nomask", "mappo_nomask_20261022_074925", "mappo_nomask_final.pt")
# EVALUATION_SEEDS = [101, 102, 103, 104, 105]
EVALUATION_SEEDS = [1001, 1002, 1003, 1004, 1005]
DEMAND_LEVELS = ["low", "normal", "high"]
UNIT_LEVELS = [10, 15, 20]

class Tracker:
    def __init__(self):
        self.prev = {}
        self.wait = {}
        self.spawn_time = {}
        self.spawned = 0
        self.depart = {}
        self.tt = {}
        self.teleports = 0
        self.teleports_upk = 0
        self.teleporting = set()   # UPK yang sedang di-teleport SUMO (keluar dari jaringan sementara)
        self.miss = {}
        self.lost = set()

class FixedTime:
    name = "fixed_time"
    env_step = False
    def setup(self, env, states):
        pass
    def act(self, env, states):
        pass


class Greedy:
    name = "greedy_preemption"
    env_step = True
    def setup(self, env, states):
        env._greedy_link_phase_mapping = {tl: gp.build_link_phase_mapping(env, tl) for tl in env.lampu_merah}
    def act(self, env, states):
        gp.apply_greedy_preemption(env)


class Mappo:
    env_step = True
    def __init__(self, name, model_path, use_mask, module, gate=False):
        self.name = name
        self.module = module
        self.model_path = model_path
        self.use_mask = use_mask
        self.mappo = None
        self.gate = gate

    def setup(self, env, states):
        if self.mappo is not None:
            return
        state_dim = {tl: len(states[tl]) for tl in env.lampu_merah}
        action_dim = {tl: len(env.action_space_marl[tl]) for tl in env.lampu_merah}
        MAPPO = importlib.import_module(self.module).MAPPO
        mappo = MAPPO(num_agents=len(env.lampu_merah), state_dim=state_dim, action_dim=action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, epochs=10, device=DEVICE)
        ckpt = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
        for tl in mappo.agent_names:
            mappo.actors[tl].load_state_dict(ckpt["actors"][tl])
            mappo.actors[tl].eval()
        mappo.critic.load_state_dict(ckpt["critic"])
        mappo.critic.eval()
        self.mappo = mappo

    def act(self, env, states):
        for tl in env.lampu_merah:
            env.maintain_preemption(tl, env.get_emv_phases(tl))
            if env.transition_active[tl]:
                continue
            if env.preemption_active[tl] and env.green_elapsed[tl] < MIN_GREEN_DURATION:
                continue
            if self.gate and not env.get_emv_phases(tl):
                continue
            x = torch.tensor(states[tl], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = self.mappo.actors[tl](x)
            if self.use_mask:
                mask = torch.tensor(env.get_action_mask(tl), dtype=torch.bool, device=DEVICE).unsqueeze(0)
                logits = logits.masked_fill(~mask, -1e9)
            action_id = int(torch.argmax(logits, dim=-1).item())
            if action_id == 0:
                continue
            phase = action(env.action_space_marl, tl, action_id)
            if phase is None or phase == "HOLD":
                continue
            env.execute_action(tl, phase, hold=True)

def make_controllers():
    return {
        "fixed_time": FixedTime(),
        "greedy_preemption": Greedy(),
        "mappo_nomask": Mappo("mappo_nomask", MODEL_NOMASK, False, "training.mappo_nomask"),
        "mappo_masked": Mappo("mappo_masked", MODEL_MASKED, True, "training.mappo"),
        "mappo_nomask_gated": Mappo("mappo_nomask_gated", MODEL_NOMASK, False, "training.mappo_nomask", gate=True),
    }

def spawn_fire_events(env, tk):
    now = traci.simulation.getTime()
    for ev in env.daftar_fire_event:
        if ev["status"] or now < ev["time_fire_event"]:
            continue
        for i in range(ev["vehicle_number"]):
            vid = f"fire_truck_{env.counter}"
            ok = env.spawnFireTruck(entry_edge=ev["entry_edges"][i], destination_edge=ev["destination_edge"], route_id=f"route_fire_truck_{env.counter}", vehicle_id=vid)
            if ok is not False:
                env.active_fire_trucks.append({"vehicle_id": vid, "destination_edge": ev["destination_edge"], "waktu_spawn": now})
                tk.prev[vid] = 0.0
                tk.wait[vid] = 0.0
                tk.spawn_time[vid] = now
                tk.spawned += 1
            env.counter += 1
        ev["status"] = True

def record_upk_waiting(env, tk):
    ids = set(traci.vehicle.getIDList())
    for t in env.active_fire_trucks:
        vid = t["vehicle_id"]
        if vid not in ids:
            continue
        try:
            cur = float(traci.vehicle.getWaitingTime(vid))
        except Exception:
            continue
        d = cur - tk.prev.get(vid, 0.0)
        if d > 0:
            tk.wait[vid] += d
        tk.prev[vid] = cur

def update_upk(env, tk):
    now = traci.simulation.getTime()
    starting = set(traci.simulation.getStartingTeleportIDList())
    ending = set(traci.simulation.getEndingTeleportIDList())
    arrived_now = set(traci.simulation.getArrivedIDList())
    tk.teleporting |= starting
    on_road = set(traci.vehicle.getIDList())
    pending = None
    remaining = []

    for t in env.active_fire_trucks:
        v = t["vehicle_id"]

        if v in on_road:
            tk.teleporting.discard(v)
            tk.miss.pop(v, None)
            if traci.vehicle.getRoadID(v) == t["destination_edge"]:
                env.buat_avg_arrival_time_upk[v] = now - t["waktu_spawn"]
                traci.vehicle.remove(v)
                continue
            remaining.append(t)
            continue

        if v in arrived_now:
            env.buat_avg_arrival_time_upk[v] = now - t["waktu_spawn"]
            continue

        if pending is None:
            pending = set(traci.simulation.getPendingVehicles())
        if v in pending or v in tk.teleporting:
            remaining.append(t)
            continue

        tk.miss[v] = tk.miss.get(v, 0) + 1
        if tk.miss[v] > UPK_MISS_LIMIT:
            tk.lost.add(v)
        else:
            remaining.append(t)

    tk.teleporting -= ending
    env.active_fire_trucks = remaining


def track_non_upk_travel_time(tk):
    now = traci.simulation.getTime()
    if now > MAX_SIMULATION_TIME:
        return
    for v in traci.simulation.getDepartedIDList():
        if v.startswith("fire_truck_"):
            continue
        try:
            if traci.vehicle.getVehicleClass(v) == "emergency":
                continue
        except Exception:
            pass
        tk.depart[v] = now
    for v in traci.simulation.getArrivedIDList():
        if v in tk.depart:
            tk.tt[v] = now - tk.depart.pop(v)


def track_teleports(tk):
    now = traci.simulation.getTime()
    for v in traci.simulation.getStartingTeleportIDList():
        is_upk = v.startswith("fire_truck_")
        if is_upk:
            tk.teleports_upk += 1
        if now <= MAX_SIMULATION_TIME:
            tk.teleports += 1


def collect_non_upk(env):
    if traci.simulation.getTime() > MAX_SIMULATION_TIME:
        return
    for tl in env.lampu_merah:
        env.buat_avg_queue_length_non_upk[tl].append(env.hitung_queue_length_non_upk(tl))
        env.buat_avg_waiting_time_non_upk[tl].append(env.hitung_waiting_time_non_upk(tl))

def upk_all_done(env):
    return all(ev["status"] for ev in env.daftar_fire_event) and not env.active_fire_trucks

def describe_remaining_upk(env, tk):
    on_road = set(traci.vehicle.getIDList())
    pending = None
    out = []
    for t in env.active_fire_trucks:
        v = t["vehicle_id"]
        if v in on_road:
            try:
                out.append(f"{v} JALAN edge={traci.vehicle.getRoadID(v)} tujuan={t['destination_edge']} "
                           f"v={traci.vehicle.getSpeed(v):.1f}m/s wait={traci.vehicle.getWaitingTime(v):.0f}s")
            except Exception:
                out.append(f"{v} JALAN (gagal baca)")
        else:
            if pending is None:
                pending = set(traci.simulation.getPendingVehicles())
            out.append(f"{v} {'MENUNGGU-INSERT' if v in pending else ('TELEPORT' if v in tk.teleporting else 'HILANG?')}")
    return out

def should_stop(env):
    now = traci.simulation.getTime()
    if now >= MAX_EXTENDED_TIME:
        return True
    return now >= MAX_SIMULATION_TIME and upk_all_done(env)

def run_episode(ctrl, seed, demand, units):
    t0 = time.time()
    start = datetime.now()
    env = TrafficEnvironment(demand_level=demand, jumlah_unit=units)
    tk = Tracker()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            states, _ = env.reset(seed=seed)
        ctrl.setup(env, states)
        interval = env.steps_per_decision
        steps = 0
        stalled = False
        next_report = MAX_SIMULATION_TIME
        while not should_stop(env):
            spawn_fire_events(env, tk)
            ctrl.act(env, states)
            for _ in range(interval):
                t_before = traci.simulation.getTime()
                if ctrl.env_step:
                    env.simulation_step()
                    if hasattr(env, "update_emv_tracking"):
                        env.update_emv_tracking()
                else:
                    traci.simulationStep()
                if traci.simulation.getTime() <= t_before:
                    stalled = True
                    break
                record_upk_waiting(env, tk)
                update_upk(env, tk)
                track_non_upk_travel_time(tk)
                track_teleports(tk)
                collect_non_upk(env)
                steps += 1
                if should_stop(env):
                    break
            now = traci.simulation.getTime()
            if now >= next_report:
                print(f"       ... t={now:.0f}s | UPK belum sampai={len(env.active_fire_trucks)} | "
                      f"kendaraan di jaringan={traci.vehicle.getIDCount()} | belum masuk jaringan={len(traci.simulation.getPendingVehicles())} | "
                      f"{time.time() - t0:.0f}s berjalan")
                for line in describe_remaining_upk(env, tk):
                    print(f"           {line}")
                next_report += 1800
            if stalled:
                print(f"[WARN] waktu SUMO berhenti di {traci.simulation.getTime():.0f}s. Periksa <end> di banjarmasintengah.sumocfg.")
                break
            if ctrl.env_step:
                states = state(env.lampu_merah)
        sim_time = traci.simulation.getTime()
        arrival = dict(env.buat_avg_arrival_time_upk)
        wait = {v: w for v, w in tk.wait.items() if v in arrival}

        cens_tt = dict(arrival)
        cens_wt = dict(wait)
        for v, t_spawn in tk.spawn_time.items():
            if v not in cens_tt:
                cens_tt[v] = sim_time - t_spawn
                cens_wt[v] = tk.wait.get(v, 0.0)

        queue_tl = queueLengthNonEMV(env.buat_avg_queue_length_non_upk)
        wait_tl = waitingTimeNonEMV(env.buat_avg_waiting_time_non_upk)
        planned = sum(e["vehicle_number"] for e in env.daftar_fire_event)
        row = {
            "method": ctrl.name, "seed": seed, "demand_level": demand, "jumlah_unit": units,
            "upk_planned": planned, "upk_spawned": tk.spawned, "upk_arrived": len(arrival),
            "upk_arrival_rate": len(arrival) / max(tk.spawned, 1),
            "upk_finished_all": int(len(arrival) == tk.spawned),
            "upk_lost": len(tk.lost),
            "non_upk_arrived": len(tk.tt), "non_upk_unfinished": len(tk.depart),
            "teleports_total": tk.teleports, "teleports_upk": tk.teleports_upk,
            "tt_upk": arrivalTimeUPK(arrival) if arrival else 0.0, "wt_upk": waitingTimeUPK(wait) if wait else 0.0,
            "tt_upk_cens": arrivalTimeUPK(cens_tt) if cens_tt else 0.0,
            "wt_upk_cens": waitingTimeUPK(cens_wt) if cens_wt else 0.0,
            "tt_non_upk": travelTimeNonEMV(tk.tt), "ql_non_upk": float(np.mean(list(queue_tl.values()))), "wt_non_upk": float(np.mean(list(wait_tl.values()))),
            "start_datetime": start.strftime("%Y-%m-%d %H:%M:%S"), "wall_clock_seconds": round(time.time() - t0, 2), "simulation_time_seconds": sim_time, "total_steps": steps,
            "stalled": int(stalled),
        }
        key = {"method": ctrl.name, "seed": seed, "demand_level": demand, "jumlah_unit": units}
        upk_rows = [
            {**key, "vehicle_id": v, "arrived": int(v in arrival), "travel_time": cens_tt[v], "waiting_time": cens_wt.get(v, 0.0)}
            for v in sorted(cens_tt)
        ]
        tl_rows = [{**key, "traffic_light": tl, "queue_length": queue_tl[tl], "waiting_time": wait_tl[tl]} for tl in queue_tl]
        return row, upk_rows, tl_rows
    finally:
        env.close()


def append_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def finished_runs(method, units):
    path = os.path.join(RESULT_ROOT, method, f"runs_{units}.csv")
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {(int(r["seed"]), r["demand_level"], int(r["jumlah_unit"])) for r in csv.DictReader(f)}


def run_and_save(ctrl, seed, demand, units):
    if (seed, demand, units) in finished_runs(ctrl.name, units):
        print(f"[skip] {ctrl.name} seed={seed} demand={demand} unit={units}")
        return
    print(f"[run ] {ctrl.name} seed={seed} demand={demand} unit={units}")
    try:
        row, upk_rows, tl_rows = run_episode(ctrl, seed, demand, units)
    except Exception:
        print(f"[FAIL] {ctrl.name} seed={seed} demand={demand} unit={units}")
        traceback.print_exc()
        return
    d = os.path.join(RESULT_ROOT, ctrl.name)
    append_csv(os.path.join(d, f"upk_detail_{units}.csv"), upk_rows)
    append_csv(os.path.join(d, f"per_traffic_light_{units}.csv"), tl_rows)
    append_csv(os.path.join(d, f"runs_{units}.csv"), [row])
    print(f"       TT_UPK={row['tt_upk']:.2f} (cens {row['tt_upk_cens']:.2f}) WT_UPK={row['wt_upk']:.2f} (cens {row['wt_upk_cens']:.2f}) TT_non={row['tt_non_upk']:.2f} QL_non={row['ql_non_upk']:.3f} WT_non={row['wt_non_upk']:.3f} | UPK {row['upk_arrived']}/{row['upk_spawned']} | end={row['simulation_time_seconds']:.0f}s | {row['wall_clock_seconds']}s")


def run_batch(method, units, demands=None, seeds=None):
    ctrl = make_controllers()[method]
    demands = demands or DEMAND_LEVELS
    seeds = seeds or EVALUATION_SEEDS
    total = len(demands) * len(seeds)
    print(f"{method} | unit {units} | {total} simulasi")
    t0 = time.time()
    n = 0
    for demand in demands:
        for seed in seeds:
            n += 1
            print(f"--- {n}/{total}")
            run_and_save(ctrl, seed, demand, units)
    print(f"Selesai {method} unit {units} dalam {time.time() - t0:.1f} detik. Hasil di {os.path.join(RESULT_ROOT, method)}")


def cli(method, units):
    ap = argparse.ArgumentParser()
    ap.add_argument("--demands", nargs="+", default=DEMAND_LEVELS, choices=DEMAND_LEVELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=EVALUATION_SEEDS)
    a = ap.parse_args()
    run_batch(method, units, a.demands, a.seeds)


def cli_all(method):
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", type=int, default=UNIT_LEVELS, choices=UNIT_LEVELS)
    ap.add_argument("--demands", nargs="+", default=DEMAND_LEVELS, choices=DEMAND_LEVELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=EVALUATION_SEEDS)
    a = ap.parse_args()
    t0 = time.time()
    for units in a.units:
        run_batch(method, units, a.demands, a.seeds)
    print(f"SEMUA SELESAI {method} dalam {time.time() - t0:.1f} detik")