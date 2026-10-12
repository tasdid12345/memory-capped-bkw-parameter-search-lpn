"""Pilot-searched BKW parameters vs textbook BKW parameters on simulated LPN (n=24, tau=0.125)."""
import itertools, json, math, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, TAU, M_MAX, TL = 24, 0.125, 2 ** 16, 5.0   # secret length, noise, samples per instance, per-run wall limit (s)
SEEDS = (42, 43, 44)
N_PILOT, N_TEST = 10, 200


def parity(v):  # parity of the bits of each int64 in v
    for sh in (16, 8, 4, 2, 1):
        v = v ^ (v >> sh)
    return v & 1


def make_instance(seed):
    """Secret s and M_MAX samples packed as int64: bits 0..N-1 = a, bit N = noisy label."""
    rng = np.random.default_rng(seed)
    s = int(rng.integers(0, 2 ** N))
    x = rng.integers(0, 2 ** N, M_MAX, dtype=np.int64)
    y = parity(x & s) ^ (rng.random(M_MAX) < TAU).astype(np.int64)
    return s, x | (y << N)


def wht(f):  # fast Walsh-Hadamard transform
    h = 1
    while h < len(f):
        f = f.reshape(-1, 2, h)
        f = np.stack((f[:, 0] + f[:, 1], f[:, 0] - f[:, 1]), 1).reshape(-1)
        h *= 2
    return f


def bkw(z, sizes):
    """Plain BKW. sizes = block sizes summing to N; all but the last block are zeroed by
    table-based reduction, then blocks are recovered last-to-first by Walsh-Hadamard voting
    (block j is voted on the batch that has had blocks < j zeroed). Returns (s_hat, stored vectors)."""
    offs = [int(o) for o in np.cumsum([0] + sizes[:-1])]
    t = len(sizes) - 1
    layers, stored = [z], 0
    for j in range(t):
        z = layers[-1]
        v = (z >> offs[j]) & ((1 << sizes[j]) - 1)
        idx = np.flatnonzero(v)
        _, first = np.unique(v[idx], return_index=True)
        rep = idx[first]                       # first sample seen per non-zero block value = stored table entry
        table = np.zeros(1 << sizes[j], np.int64)
        table[v[rep]] = z[rep]
        out = np.where(v > 0, z ^ table[v], z)
        keep = np.ones(len(z), bool)
        keep[rep] = False                      # stored vectors are consumed
        layers.append(out[keep])
        stored += len(rep)
    s_hat = 0
    for j in range(t, -1, -1):
        z = layers[j]
        y = ((z >> N) & 1) ^ parity(z & s_hat)  # remove contribution of already-recovered bits
        v = (z >> offs[j]) & ((1 << sizes[j]) - 1)
        f = wht(np.bincount(v, weights=1.0 - 2.0 * y, minlength=1 << sizes[j]))
        s_hat |= int(np.argmax(f)) << offs[j]
    return s_hat, stored


def nominal_mem(sizes):  # table capacity (vectors) of a configuration
    return sum((1 << b) - 1 for b in sizes[:-1])


def evaluate(insts, sizes, m, early=False):
    """Run BKW on instances; success needs the exact secret within the wall-clock limit."""
    ok, times, stored = [], [], []
    for i, (s, z) in enumerate(insts):
        t0 = time.perf_counter()
        s_hat, st = bkw(z[:m], sizes)
        dt = time.perf_counter() - t0
        ok.append(s_hat == s and dt <= TL); times.append(dt); stored.append(st)
        if early and i == 2 and not any(ok):   # prune hopeless configs in the pilot
            break
    return ok, times, stored


def main():
    # Baseline: textbook rule, a = ceil(log2 n) equal-ish blocks, fixed sample count
    a = math.ceil(math.log2(N)); b = math.ceil(N / a)
    base_sizes = [b] * (a - 1) + [N - b * (a - 1)]
    base_m = M_MAX
    cap = nominal_mem(base_sizes)
    print(f"baseline blocks={base_sizes} m={base_m} memory cap={cap} vectors", flush=True)

    # Improved: search on pilot instances (fixed seeds, disjoint from test seeds)
    pilot = [make_instance(900000 + i) for i in range(N_PILOT)]
    results = []
    for t in range(1, 5):
        for red in itertools.combinations_with_replacement(range(1, 9), t):  # unequal, non-decreasing sizes
            r = N - sum(red)
            sizes = list(red) + [r]
            if not 1 <= r <= 18 or nominal_mem(sizes) > cap:
                continue
            for m in (2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16):
                ok, times, _ = evaluate(pilot, sizes, m, early=True)
                results.append((-sum(ok) / N_PILOT, float(np.mean(times)), sizes, m))
    if not results:
        sys.exit("no feasible configuration")
    results.sort(key=lambda x: (x[0], x[1]))
    neg_rate, ptime, imp_sizes, imp_m = results[0]
    print(f"searched {len(results)} configs; best blocks={imp_sizes} m={imp_m} pilot success={-neg_rate:.2f}", flush=True)
    assert nominal_mem(imp_sizes) <= cap

    # Held-out evaluation: fresh instances per seed, same instances for both methods
    out = {k: {"succ": [], "time": [], "stored": []} for k in ("base", "imp")}
    for seed in SEEDS:
        test = [make_instance(seed * 10000 + i) for i in range(N_TEST)]
        for key, sizes, m in (("base", base_sizes, base_m), ("imp", imp_sizes, imp_m)):
            ok, times, stored = evaluate(test, sizes, m)
            out[key]["succ"].append(float(np.mean(ok)))
            out[key]["time"].append(float(np.mean(times)))
            out[key]["stored"].append(int(max(stored)))
            print(f"seed {seed} {key}: recovery={np.mean(ok):.3f} time={np.mean(times):.4f}s", flush=True)

    bs, im = out["base"], out["imp"]
    metrics = {
        "baseline_score": float(np.mean(bs["succ"])), "improved_score": float(np.mean(im["succ"])),
        "baseline_score_std": float(np.std(bs["succ"])), "improved_score_std": float(np.std(im["succ"])),
        "baseline_mean_time_s": float(np.mean(bs["time"])), "improved_mean_time_s": float(np.mean(im["time"])),
        "baseline_peak_stored_vectors": float(max(bs["stored"])), "improved_peak_stored_vectors": float(max(im["stored"])),
        "memory_cap_vectors": float(cap), "pilot_success_of_chosen": float(-neg_rate),
        "n_configs_searched": float(len(results)), "improved_num_samples": float(imp_m),
        "baseline_num_samples": float(base_m),
    }
    metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
    if not all(np.isfinite(v) for v in metrics.values()):
        sys.exit("non-finite metrics")

    os.makedirs("figures", exist_ok=True)
    plt.bar(["baseline BKW", "searched BKW"], [metrics["baseline_score"], metrics["improved_score"]],
            yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], color=["gray", "tab:blue"])
    plt.ylabel("full-secret recovery rate (held-out)"); plt.tight_layout()
    plt.savefig("figures/recovery.png"); plt.close()

    res = {
        "question": "Under a fixed memory cap, can a pilot-run parameter search for plain BKW on small simulated LPN instances recover the secret more often than a hand-picked textbook BKW parameter rule?",
        "method": "Simulated LPN (n=24, noise 0.125). Baseline: plain BKW with ceil(log2 n) near-equal blocks and a fixed sample count. Improved: same BKW code with number of blocks, block sizes and sample count searched on pilot instances under the baseline's table-memory cap. Evaluated on 200 fresh held-out instances per seed (seeds 42, 43, 44).",
        "dataset": {"name": "Simulated LPN instances (n=24, tau=0.125)", "source_url": "simulated with numpy, no download",
                    "n_rows": N_PILOT + N_TEST, "n_train": N_PILOT, "n_test": N_TEST,
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "full-secret recovery rate on held-out instances", "higher_is_better": True,
        "baseline": f"Plain BKW, textbook rule: {a} blocks of sizes {base_sizes}, {base_m} samples.",
        "improved": f"Plain BKW with pilot-searched parameters: block sizes {imp_sizes}, {imp_m} samples, chosen by pilot success then time, table capacity <= {cap} vectors.",
        "metrics": metrics,
        "notes": "Dissection-BKW was not implemented (paper details unavailable); baseline is a stand-in textbook BKW. Rows = instances (10 pilot, 200 held-out per seed, 65536 samples each); the algorithms are deterministic, seeds change the held-out instances. Memory cap = sum of (2^b-1) over reduced blocks of the baseline; peak stored vectors = max actual table entries. The final-block Walsh-Hadamard accumulator (2^r counters) and the retained sample batches (re-generable stream) are not counted as stored vectors, which favours configurations with a large last block. Per-run wall-clock limit 5 s (overruns count as failure). Pilot set is small (10 instances), configs with 0/3 early successes are pruned.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
