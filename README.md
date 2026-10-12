# Pilot-searched vs textbook BKW parameters on simulated LPN

Motivation: the paper "Dissection-BKW" (2018) improves the Blum-Kalai-Wasserman (BKW) algorithm for LPN/LWE by choosing how the secret is split and reduced. Only a truncated abstract was available, so this experiment does not implement Dissection-BKW. It tests a simpler, related idea: whether tuning the parameters of plain BKW on pilot runs beats a hand-picked textbook rule.

Question: under a fixed memory cap, can a pilot-run parameter search for plain BKW on small simulated LPN instances recover the secret more often than a hand-picked textbook BKW parameter rule?

Dataset: simulated LPN instances (secret length n=24, noise rate 0.125) generated with numpy, no download. This is not the paper's dataset. Pilot instances (fixed seeds) are used for parameter selection; 200 fresh held-out instances per seed (42, 43, 44) are used for testing.

Baseline vs improved:
- Baseline: plain BKW with a = ceil(log2 n) near-equal blocks and a fixed number of samples (a stand-in, not the paper's method).
- Improved: the same BKW code, sample generator, memory accounting and wall-clock limit, but the number of blocks, the (possibly unequal) block sizes and the sample count are searched on pilot instances. Only configurations whose table memory is within the baseline's cap are kept. The best is chosen by pilot success rate, then by time.

Main metric: full-secret recovery rate on held-out instances. Secondary: mean runtime and peak stored vectors.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to results.json (and a chart to figures/).

## Results (latest run)

On simulated LPN instances (n=24, noise 0.125), a standard BKW with hand-chosen textbook parameters recovered the full secret on 64.5% of 200 held-out instances, while pilot-searched plain BKW reached 97.3% under the same 124-vector memory cap.

| Metric | Value |
|---|---|
| baseline_score | 0.645 |
| improved_score | 0.9733 |
| baseline_score_std | 0.0694 |
| improved_score_std | 0.017 |
| baseline_mean_time_s | 0.01248 |
| improved_mean_time_s | 0.000481 |
| baseline_peak_stored_vectors | 124 |
| improved_peak_stored_vectors | 94 |
| memory_cap_vectors | 124 |
| pilot_success_of_chosen | 1 |
| n_configs_searched | 632 |
| improved_num_samples | 1024 |
| baseline_num_samples | 65536 |
| improvement | 0.3283 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Dissection-BKW was not implemented (paper details unavailable); baseline is a stand-in textbook BKW. Rows = instances (10 pilot, 200 held-out per seed, 65536 samples each); the algorithms are deterministic, seeds change the held-out instances. Memory cap = sum of (2^b-1) over reduced blocks of the baseline; peak stored vectors = max actual table entries. The final-block Walsh-Hadamard accumulator (2^r counters) and the retained sample batches (re-generable stream) are not counted as stored vectors, which favours configurations with a large last block. Per-run wall-clock limit 5 s (overruns count as failure). Pilot set is small (10 instances), configs with 0/3 early successes are pruned.

## Reference

This project is motivated by the following paper:

> A Esser, F Heuer, R Kübler, A May, C Sohler (2018). *Dissection-bkw*. Annual International. https://link.springer.com/chapter/10.1007/978-3-319-96881-0_22

Paper author: Prof. Dr. Alexander May, Ruhr-Universität Bochum (alexander.may@rub.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
