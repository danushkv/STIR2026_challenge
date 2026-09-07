# STIR checkpoint ranking — LiteTracker streaming runtime

Merged 19 checkpoint shards from the reproduction evaluation sweep. Paired clip-level bootstrap, 10000 resamples.


## 2D — 32 clips, 234 points, 31 runs, 10000 clip-bootstrap resamples

Measured noise floor at the 8px threshold:

- points/clip 7.3, intra-clip correlation 0.403, design effect 3.54
- naive binomial SE 2.90pp → clustered SE 5.46pp
- **unpaired** comparisons need ~21.4pp to mean anything; the paired deltas below need far less.

| run | delta_avg | 95% CI | mean px | median px |
|---|---|---|---|---|
| agg_e40|i2 | 0.8137 | [0.7578, 0.8566] | 8.78 | 4.60 |
| agg_e40|i4 | 0.8111 | [0.7533, 0.8553] | 9.05 | 4.59 |
| agg_e44|i4 | 0.8103 | [0.7480, 0.8579] | 9.38 | 4.60 |
| agg_e44_match|i4 | 0.8103 | [0.7480, 0.8579] | 9.38 | 4.60 |
| agg_e44_match_fix|i4 | 0.8103 | [0.7480, 0.8579] | 9.38 | 4.60 |
| agg_e44_match_fix2|i4 | 0.8103 | [0.7480, 0.8579] | 9.38 | 4.60 |
| agg_e44_match_fix3|i4 | 0.8103 | [0.7480, 0.8579] | 9.38 | 4.60 |
| agg_e44_match_fix4|i4 | 0.8103 | [0.7480, 0.8579] | 9.38 | 4.60 |
| agg_e44|i1 | 0.8085 | [0.7484, 0.8553] | 9.75 | 4.55 |
| agg_e40|i1 | 0.8077 | [0.7471, 0.8536] | 9.79 | 4.64 |
| agg_e44|i2 | 0.8077 | [0.7450, 0.8556] | 9.43 | 4.60 |
| agg_around_thr_e30|i1 | 0.8060 | [0.7462, 0.8521] | 9.96 | 4.72 |
| agg_around_thr_e30|i2 | 0.8060 | [0.7469, 0.8516] | 9.42 | 4.57 |
| peng_baseline|i1 | 0.8051 | [0.7493, 0.8498] | 9.13 | 4.75 |
| agg_around_e30|i1 | 0.8043 | [0.7434, 0.8515] | 10.59 | 4.71 |
| agg_around_e44|i1 | 0.8034 | [0.7432, 0.8496] | 10.16 | 4.63 |
| agg_around_thr_e30|i4 | 0.8034 | [0.7435, 0.8495] | 10.16 | 4.56 |
| around_peng_e42|i1 | 0.8034 | [0.7484, 0.8465] | 9.27 | 4.91 |
| agg_e44_repro|i4 | 0.8017 | [0.7438, 0.8468] | 9.79 | 4.59 |
| around_peng_thr_e44|i1 | 0.8017 | [0.7448, 0.8461] | 9.21 | 4.66 |
| extra_thr_peng_e40|i1 | 0.8017 | [0.7399, 0.8494] | 10.10 | 4.60 |
| agg_e44_repro|i2 | 0.8009 | [0.7406, 0.8468] | 10.41 | 4.58 |
| extra_thr_e42|i1 | 0.8009 | [0.7412, 0.8461] | 10.44 | 4.64 |
| extra_e40|i2 | 0.8000 | [0.7407, 0.8457] | 9.57 | 4.77 |
| extra_e40|i4 | 0.8000 | [0.7407, 0.8457] | 9.42 | 4.77 |
| extra_thr_e42|i2 | 0.8000 | [0.7394, 0.8466] | 10.47 | 4.55 |
| extra_thr_e42|i4 | 0.8000 | [0.7404, 0.8455] | 9.58 | 4.61 |
| agg_e44_repro|i1 | 0.7991 | [0.7392, 0.8455] | 10.53 | 4.61 |
| extra_thr_e30|i1 | 0.7991 | [0.7374, 0.8465] | 10.62 | 4.76 |
| extra_e40|i1 | 0.7974 | [0.7349, 0.8447] | 10.44 | 4.75 |
| trial_e45|i1 | 0.7974 | [0.7341, 0.8464] | 212.78 | 4.70 |

**Paired vs leader (agg_e40|i2)** — same clips, same resamples:

| run | delta | 95% CI | P(beats leader) | separable? |
|---|---|---|---|---|
| agg_e40|i4 | -0.0026 | [-0.0077, +0.0000] | 0.000 | no — tie |
| agg_e44|i4 | -0.0034 | [-0.0144, +0.0052] | 0.218 | no — tie |
| agg_e44_match|i4 | -0.0034 | [-0.0144, +0.0052] | 0.218 | no — tie |
| agg_e44_match_fix|i4 | -0.0034 | [-0.0144, +0.0052] | 0.218 | no — tie |
| agg_e44_match_fix2|i4 | -0.0034 | [-0.0144, +0.0052] | 0.218 | no — tie |
| agg_e44_match_fix3|i4 | -0.0034 | [-0.0144, +0.0052] | 0.218 | no — tie |
| agg_e44_match_fix4|i4 | -0.0034 | [-0.0144, +0.0052] | 0.218 | no — tie |
| agg_e44|i1 | -0.0051 | [-0.0162, +0.0043] | 0.131 | no — tie |
| agg_e40|i1 | -0.0060 | [-0.0152, +0.0007] | 0.027 | no — tie |
| agg_e44|i2 | -0.0060 | [-0.0172, +0.0028] | 0.091 | no — tie |
| agg_around_thr_e30|i1 | -0.0077 | [-0.0203, +0.0033] | 0.077 | no — tie |
| agg_around_thr_e30|i2 | -0.0077 | [-0.0191, +0.0023] | 0.059 | no — tie |
| peng_baseline|i1 | -0.0085 | [-0.0206, +0.0052] | 0.099 | no — tie |
| agg_around_e30|i1 | -0.0094 | [-0.0232, +0.0017] | 0.045 | no — tie |
| agg_around_e44|i1 | -0.0103 | [-0.0230, +0.0009] | 0.031 | no — tie |
| agg_around_thr_e30|i4 | -0.0103 | [-0.0224, -0.0000] | 0.017 | yes |
| around_peng_e42|i1 | -0.0103 | [-0.0213, +0.0016] | 0.037 | no — tie |
| agg_e44_repro|i4 | -0.0120 | [-0.0212, -0.0039] | 0.001 | yes |
| around_peng_thr_e44|i1 | -0.0120 | [-0.0223, -0.0031] | 0.003 | yes |
| extra_thr_peng_e40|i1 | -0.0120 | [-0.0246, -0.0027] | 0.003 | yes |
| agg_e44_repro|i2 | -0.0128 | [-0.0236, -0.0047] | 0.000 | yes |
| extra_thr_e42|i1 | -0.0128 | [-0.0250, -0.0033] | 0.003 | yes |
| extra_e40|i2 | -0.0137 | [-0.0248, -0.0038] | 0.002 | yes |
| extra_e40|i4 | -0.0137 | [-0.0248, -0.0038] | 0.002 | yes |
| extra_thr_e42|i2 | -0.0137 | [-0.0270, -0.0040] | 0.001 | yes |
| extra_thr_e42|i4 | -0.0137 | [-0.0273, -0.0035] | 0.001 | yes |
| agg_e44_repro|i1 | -0.0145 | [-0.0258, -0.0052] | 0.001 | yes |
| extra_thr_e30|i1 | -0.0145 | [-0.0294, -0.0032] | 0.003 | yes |
| extra_e40|i1 | -0.0162 | [-0.0308, -0.0047] | 0.001 | yes |
| trial_e45|i1 | -0.0162 | [-0.0323, -0.0048] | 0.001 | yes |

A CI straddling 0 means that run is **not separable** from the leader. Among ties, prefer the faster / cleaner-trained model.

Endpoint error vs clip length (agg_e40|i2): slope -0.0025 px/frame, r=-0.04 over 14–2074 frames. A clearly positive slope means drift accumulates and per-frame ATA will be worse than these endpoint numbers.


## 3D — 32 clips, 227 points, 31 runs, 10000 clip-bootstrap resamples

Measured noise floor at the 4mm threshold:

- points/clip 7.1, intra-clip correlation 0.144, design effect 1.88
- naive binomial SE 3.29pp → clustered SE 4.51pp
- **unpaired** comparisons need ~17.7pp to mean anything; the paired deltas below need far less.

| run | acc_avg | 95% CI | mean mm | median mm |
|---|---|---|---|---|
| agg_e44|i4 | 0.7383 | [0.6733, 0.7919] | 6.16 | 3.01 |
| agg_e44|i2 | 0.7366 | [0.6706, 0.7907] | 6.25 | 2.95 |
| agg_e40|i4 | 0.7348 | [0.6705, 0.7885] | 6.17 | 2.92 |
| agg_e44|i1 | 0.7339 | [0.6635, 0.7906] | 6.88 | 3.06 |
| extra_thr_e42|i4 | 0.7330 | [0.6701, 0.7903] | 6.03 | 3.10 |
| extra_thr_e42|i2 | 0.7322 | [0.6691, 0.7907] | 7.79 | 3.10 |
| extra_thr_e42|i1 | 0.7313 | [0.6742, 0.7837] | 6.39 | 3.28 |
| agg_e44_repro|i4 | 0.7313 | [0.6630, 0.7881] | 16.98 | 2.99 |
| agg_e40|i2 | 0.7304 | [0.6612, 0.7853] | 6.22 | 2.96 |
| agg_e44_repro|i1 | 0.7304 | [0.6679, 0.7862] | 15.42 | 3.07 |
| extra_e40|i2 | 0.7304 | [0.6727, 0.7807] | 6.22 | 3.02 |
| agg_e44_repro|i2 | 0.7304 | [0.6644, 0.7860] | 12.32 | 2.95 |
| extra_e40|i1 | 0.7278 | [0.6680, 0.7794] | 7.36 | 3.22 |
| extra_e40|i4 | 0.7278 | [0.6701, 0.7768] | 6.29 | 3.13 |
| agg_around_thr_e30|i2 | 0.7269 | [0.6673, 0.7793] | 6.16 | 3.21 |
| trial_e45|i1 | 0.7269 | [0.6612, 0.7854] | 8.88 | 3.05 |
| agg_e40|i1 | 0.7260 | [0.6564, 0.7822] | 6.93 | 3.05 |
| extra_thr_peng_e40|i1 | 0.7233 | [0.6590, 0.7791] | 6.46 | 3.30 |
| agg_around_thr_e30|i4 | 0.7225 | [0.6538, 0.7791] | 12.76 | 3.49 |
| extra_thr_e30|i1 | 0.7207 | [0.6619, 0.7735] | 6.16 | 3.43 |
| agg_around_e44|i1 | 0.7172 | [0.6475, 0.7791] | 21.79 | 3.43 |
| around_peng_e42|i1 | 0.7172 | [0.6498, 0.7728] | 7.09 | 3.41 |
| agg_around_thr_e30|i1 | 0.7137 | [0.6532, 0.7665] | 6.82 | 3.41 |
| agg_around_e30|i1 | 0.7093 | [0.6425, 0.7669] | 7.85 | 3.44 |
| around_peng_thr_e44|i1 | 0.7084 | [0.6372, 0.7681] | 7.96 | 3.38 |
| peng_baseline|i1 | 0.7048 | [0.6365, 0.7647] | 6.49 | 3.56 |
| agg_e44_match_fix4|i4 | 0.7031 | [0.6436, 0.7608] | 19.31 | 3.14 |
| agg_e44_match_fix2|i4 | 0.6907 | [0.6339, 0.7510] | 30.82 | 3.18 |
| agg_e44_match_fix3|i4 | 0.6907 | [0.6339, 0.7510] | 30.82 | 3.18 |
| agg_e44_match|i4 | 0.6863 | [0.6306, 0.7468] | 31.09 | 3.18 |
| agg_e44_match_fix|i4 | 0.6828 | [0.6266, 0.7385] | 7.27 | 3.42 |
| _CONTROL (never moved)_ | 0.6115 | — | 8.99 | 5.09 |

**Paired vs leader (agg_e44|i4)** — same clips, same resamples:

| run | delta | 95% CI | P(beats leader) | separable? |
|---|---|---|---|---|
| agg_e44|i2 | -0.0018 | [-0.0076, +0.0028] | 0.193 | no — tie |
| agg_e40|i4 | -0.0035 | [-0.0183, +0.0138] | 0.300 | no — tie |
| agg_e44|i1 | -0.0044 | [-0.0170, +0.0043] | 0.175 | no — tie |
| extra_thr_e42|i4 | -0.0053 | [-0.0268, +0.0182] | 0.324 | no — tie |
| extra_thr_e42|i2 | -0.0062 | [-0.0270, +0.0162] | 0.301 | no — tie |
| extra_thr_e42|i1 | -0.0070 | [-0.0277, +0.0147] | 0.255 | no — tie |
| agg_e44_repro|i4 | -0.0070 | [-0.0251, +0.0069] | 0.161 | no — tie |
| agg_e40|i2 | -0.0079 | [-0.0238, +0.0087] | 0.152 | no — tie |
| agg_e44_repro|i1 | -0.0079 | [-0.0267, +0.0091] | 0.182 | no — tie |
| extra_e40|i2 | -0.0079 | [-0.0230, +0.0057] | 0.120 | no — tie |
| agg_e44_repro|i2 | -0.0079 | [-0.0271, +0.0058] | 0.140 | no — tie |
| extra_e40|i1 | -0.0106 | [-0.0267, +0.0045] | 0.075 | no — tie |
| extra_e40|i4 | -0.0106 | [-0.0276, +0.0037] | 0.070 | no — tie |
| agg_around_thr_e30|i2 | -0.0115 | [-0.0273, +0.0045] | 0.072 | no — tie |
| trial_e45|i1 | -0.0115 | [-0.0279, +0.0070] | 0.100 | no — tie |
| agg_e40|i1 | -0.0123 | [-0.0262, +0.0012] | 0.034 | no — tie |
| extra_thr_peng_e40|i1 | -0.0150 | [-0.0366, +0.0058] | 0.070 | no — tie |
| agg_around_thr_e30|i4 | -0.0159 | [-0.0344, +0.0010] | 0.032 | no — tie |
| extra_thr_e30|i1 | -0.0176 | [-0.0372, +0.0009] | 0.028 | no — tie |
| agg_around_e44|i1 | -0.0211 | [-0.0392, +0.0000] | 0.025 | no — tie |
| around_peng_e42|i1 | -0.0211 | [-0.0407, -0.0030] | 0.010 | yes |
| agg_around_thr_e30|i1 | -0.0247 | [-0.0486, -0.0041] | 0.008 | yes |
| agg_around_e30|i1 | -0.0291 | [-0.0511, -0.0090] | 0.003 | yes |
| around_peng_thr_e44|i1 | -0.0300 | [-0.0559, -0.0071] | 0.004 | yes |
| peng_baseline|i1 | -0.0335 | [-0.0549, -0.0131] | 0.001 | yes |
| agg_e44_match_fix4|i4 | -0.0352 | [-0.0595, -0.0069] | 0.008 | yes |
| agg_e44_match_fix2|i4 | -0.0476 | [-0.0740, -0.0148] | 0.002 | yes |
| agg_e44_match_fix3|i4 | -0.0476 | [-0.0740, -0.0148] | 0.002 | yes |
| agg_e44_match|i4 | -0.0520 | [-0.0815, -0.0174] | 0.002 | yes |
| agg_e44_match_fix|i4 | -0.0555 | [-0.0792, -0.0284] | 0.000 | yes |

A CI straddling 0 means that run is **not separable** from the leader. Among ties, prefer the faster / cleaner-trained model.

Endpoint error vs clip length (agg_e44|i4): slope -0.0004 mm/frame, r=-0.02 over 14–2074 frames. A clearly positive slope means drift accumulates and per-frame ATA will be worse than these endpoint numbers.


## Drift and visibility (measurable without 2026 GT)

| run | cycle mean px | cycle/endpoint | invisible % | flicker/100f | AJ pred-vis | AJ forced-vis | gain if forced |
|---|---|---|---|---|---|---|---|
| agg_around_e30|i1 | 5.91 | 0.56x | 4.64 | 0.19 | 0.5489 | 0.5399 | -0.0089 |
| agg_around_e44|i1 | 5.92 | 0.58x | 6.62 | 0.41 | 0.5455 | 0.5394 | -0.0061 |
| agg_around_thr_e30|i1 | 5.12 | 0.51x | 4.39 | 0.20 | 0.5518 | 0.5444 | -0.0074 |
| agg_around_thr_e30|i2 | 4.43 | 0.47x | 5.04 | 0.26 | 0.5451 | 0.5453 | +0.0002 |
| agg_around_thr_e30|i4 | 4.94 | 0.49x | 5.11 | 0.29 | 0.5420 | 0.5423 | +0.0003 |
| agg_e40|i1 | 6.59 | 0.67x | 5.69 | 0.30 | 0.5561 | 0.5433 | -0.0128 |
| agg_e40|i2 | 5.25 | 0.60x | 5.47 | 0.38 | 0.5521 | 0.5498 | -0.0024 |
| agg_e40|i4 | 5.71 | 0.63x | 5.71 | 0.39 | 0.5521 | 0.5460 | -0.0061 |
| agg_e44|i1 | 5.28 | 0.54x | 6.74 | 0.37 | 0.5472 | 0.5410 | -0.0062 |
| agg_e44|i2 | 5.03 | 0.53x | 6.24 | 0.39 | 0.5539 | 0.5414 | -0.0125 |
| agg_e44|i4 | 5.71 | 0.61x | 6.48 | 0.58 | 0.5513 | 0.5438 | -0.0075 |
| agg_e44_match|i4 | 5.71 | 0.61x | 6.48 | 0.58 | 0.5513 | 0.5438 | -0.0075 |
| agg_e44_match_fix|i4 | 5.71 | 0.61x | 6.48 | 0.58 | 0.5513 | 0.5438 | -0.0075 |
| agg_e44_match_fix2|i4 | 5.71 | 0.61x | 6.48 | 0.58 | 0.5513 | 0.5438 | -0.0075 |
| agg_e44_match_fix3|i4 | 5.71 | 0.61x | 6.48 | 0.58 | 0.5513 | 0.5438 | -0.0075 |
| agg_e44_match_fix4|i4 | 5.71 | 0.61x | 6.48 | 0.58 | 0.5513 | 0.5438 | -0.0075 |
| agg_e44_repro|i1 | 5.98 | 0.57x | 4.11 | 0.42 | 0.5442 | 0.5337 | -0.0105 |
| agg_e44_repro|i2 | 5.91 | 0.57x | 4.67 | 0.52 | 0.5423 | 0.5361 | -0.0062 |
| agg_e44_repro|i4 | 5.90 | 0.60x | 4.68 | 0.64 | 0.5461 | 0.5366 | -0.0096 |
| around_peng_e42|i1 | 6.50 | 0.70x | 9.17 | 0.79 | 0.5391 | 0.5407 | +0.0017 |
| around_peng_thr_e44|i1 | 5.76 | 0.63x | 5.46 | 0.37 | 0.5449 | 0.5398 | -0.0051 |
| extra_e40|i1 | 5.72 | 0.55x | 4.32 | 0.26 | 0.5459 | 0.5309 | -0.0150 |
| extra_e40|i2 | 6.90 | 0.72x | 4.89 | 0.21 | 0.5443 | 0.5333 | -0.0111 |
| extra_e40|i4 | 5.77 | 0.61x | 4.46 | 0.29 | 0.5406 | 0.5333 | -0.0074 |
| extra_thr_e30|i1 | 4.72 | 0.44x | 3.60 | 0.53 | 0.5450 | 0.5346 | -0.0104 |
| extra_thr_e42|i1 | 5.66 | 0.54x | 4.10 | 0.24 | 0.5441 | 0.5359 | -0.0082 |
| extra_thr_e42|i2 | 4.88 | 0.47x | 4.44 | 0.29 | 0.5411 | 0.5361 | -0.0050 |
| extra_thr_e42|i4 | 4.68 | 0.49x | 4.77 | 0.34 | 0.5408 | 0.5367 | -0.0041 |
| extra_thr_peng_e40|i1 | 3.94 | 0.39x | 4.99 | 0.24 | 0.5444 | 0.5381 | -0.0063 |
| peng_baseline|i1 | 4.81 | 0.53x | 1.83 | 0.15 | 0.5430 | 0.5390 | -0.0040 |
| trial_e45|i1 | 208.41 | 0.98x | 5.47 | 0.32 | 0.5405 | 0.5331 | -0.0073 |

`gain if forced` > 0 means the visibility head is COSTING you AJ: setting `visibs=True` in the wrapper is legal and strictly better. An invisible rate under ~1% means the head is inert (AJ ≈ ATA), which is the safe failure mode.

Lowest `cycle mean px` at equal endpoint error = the model that actually tracks, rather than one the nearest-neighbour match rescues. Read cycle error comparatively, not as an error budget.
