"""Do the walker behaviours produce a crossing-prediction task that is actually
hard? Simulates each behaviour kinematically (no CARLA) and checks the mix.
"""
import sys
import random
import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

import yaml
from carla_tools.walker_behavior import (
    ALL_KINDS, APPROACH_ABORT, CROSS_HESITATE, CROSS_RUN, CROSS_STEADY, STOP_MIDWAY,
    BehaviourParams, CrossingBehaviour, sample_behaviour,
)

DT = 0.1
CFG = yaml.safe_load(open(r"E:\Project Phase 1\configs\scenarios.yaml"))["vru_behaviour"]
CORRIDOR_EDGE = 4.0 - 1.75     # walker starts 4 m out; corridor half-width 1.75 m
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)


def simulate(params, n=60):
    """Runs a behaviour open-loop, returning the distance trace."""
    b = CrossingBehaviour(params)
    d, trace = 0.0, []
    for _ in range(n):
        speed, sign = b.step(DT, d)
        d += speed * sign * DT
        trace.append(d)
    return np.array(trace), b


print("1. Each behaviour does what its name says")
base = dict(speed_ms=1.4, start_delay_s=0.3, kerb_distance_m=2.0,
            hesitate_s=1.0, stop_at_m=3.0, run_multiplier=2.0)
traces = {}
for kind in ALL_KINDS:
    tr, b = simulate(BehaviourParams(kind=kind, **base))
    traces[kind] = tr
    print(f"  {kind:<16} final={tr[-1]:5.2f} m  max={tr.max():5.2f} m  end state={b.state}")

check("steady crosser gets furthest", traces[CROSS_STEADY][-1] > CORRIDOR_EDGE)
check("runner beats steady walker", traces[CROSS_RUN][-1] > traces[CROSS_STEADY][-1])
check("hesitater still crosses, just later",
      traces[CROSS_HESITATE][-1] > CORRIDOR_EDGE)
check("hesitater lags the steady walker",
      traces[CROSS_HESITATE][-1] < traces[CROSS_STEADY][-1])
check("aborter returns to its start", abs(traces[APPROACH_ABORT][-1]) < 0.2,
      f"{traces[APPROACH_ABORT][-1]:.2f} m")
check("aborter did approach first", traces[APPROACH_ABORT].max() > 1.5,
      f"peak {traces[APPROACH_ABORT].max():.2f} m")
check("stop_midway halts before the far side",
      traces[STOP_MIDWAY][-1] < traces[CROSS_STEADY][-1])

print("\n2. The abort is genuinely ambiguous early on")
h = traces[CROSS_HESITATE]
a = traces[APPROACH_ABORT]
first_diff = int(np.argmax(np.abs(h - a) > 0.05))
print(f"  hesitate and abort traces stay identical for {first_diff} ticks "
      f"({first_diff * DT:.1f} s)")
check("indistinguishable for at least a second", first_diff * DT >= 1.0)

print("\n3. Sampling produces a workable class balance")
rng = random.Random(0)
kinds = [sample_behaviour(rng, CFG, road_edge_m=CORRIDOR_EDGE).kind for _ in range(4000)]
crossers = sum(k in (CROSS_STEADY, CROSS_HESITATE, CROSS_RUN) for k in kinds)
frac = crossers / len(kinds)
for k in ALL_KINDS:
    print(f"  {k:<16} {kinds.count(k) / len(kinds):.3f}")
print(f"  -> crossing fraction {frac:.3f}")
check("class balance near 60/40", 0.5 < frac < 0.7, f"{frac:.3f}")
check("all five kinds occur", set(kinds) == set(ALL_KINDS))

print("\n4. Speeds vary as configured")
speeds = [sample_behaviour(rng, CFG, road_edge_m=CORRIDOR_EDGE).speed_ms for _ in range(2000)]
print(f"  speed range {min(speeds):.2f} - {max(speeds):.2f} m/s, mean {np.mean(speeds):.2f}")
check("spans the configured range", min(speeds) < 1.0 and max(speeds) > 2.3)
check("not the old fixed 1.2", np.std(speeds) > 0.3, f"std {np.std(speeds):.2f}")

print("\n5. THE POINT: a constant-velocity predictor can no longer ace this")
# At the moment of the kerb decision, extrapolate at current velocity and ask
# whether the walker ends up past the corridor edge. Compare to what happened.
rng = random.Random(7)
correct = tp = fp = fn = tn = 0
for _ in range(3000):
    p = sample_behaviour(rng, CFG, road_edge_m=CORRIDOR_EDGE)
    trace, _ = simulate(p, n=80)
    # Predict at the tick the walker first reaches the kerb decision point.
    idx = np.argmax(trace >= p.kerb_distance_m)
    if trace[idx] < p.kerb_distance_m:
        continue                       # never got there
    v = (trace[idx] - trace[max(idx - 3, 0)]) / (3 * DT)
    predicted = (trace[idx] + v * 3.0) > CORRIDOR_EDGE     # 3 s constant-velocity
    actual = trace.max() > CORRIDOR_EDGE
    correct += predicted == actual
    tp += predicted and actual
    fp += predicted and not actual
    fn += (not predicted) and actual
    tn += (not predicted) and (not actual)
n = tp + fp + fn + tn
acc = correct / n
print(f"  constant-velocity accuracy at the kerb: {acc:.3f}  "
      f"(TP={tp} FP={fp} FN={fn} TN={tn})")
always_yes = (tp + fn) / n
print(f"  an 'always crosses' predictor would score: {always_yes:.3f}")
check("constant velocity is no longer near-perfect", acc < 0.95, f"{acc:.3f}")
check("it is beaten badly enough to leave headroom", acc < 0.9, f"{acc:.3f}")
check("always-yes is not a winning strategy", always_yes < 0.75, f"{always_yes:.3f}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
