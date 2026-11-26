#!/usr/bin/env python3
"""Independent standard-library float64 / exact oracle; imports no port code.
Matrices flatten feature-major, case-fastest. Special floats use strict JSON strings.
Finite-difference tests in the C# harness also use independent float64 functions.
"""
import json
import math
import struct
from pathlib import Path


def flat(cases):
    return [row[c] for c in range(len(cases[0])) for row in cases]


def log(x):
    return math.log(x) if x > 0 else -math.inf if x == 0 else math.nan


def softmax(z):
    # Finite max is order-independent. IEEE singular cases intentionally remain NaN.
    if any(math.isnan(x) for x in z) or max(z) in (-math.inf, math.inf):
        return [math.nan] * len(z)
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return [x / sum(e) for x in e]


def cost(cases, labels, k):
    logs, correct, top = [], [], []
    for p, label in zip(cases, labels):
        lp = p[label]
        m = p[0]
        for v in p[1:]:
            m = v if v > m else m
        bigger = sum(x > lp for x in p)
        equal = sum(x == lp for x in p)
        logs.append(log(lp))
        correct.append(1 / equal if lp == m else 0)
        slots = k - bigger
        top.append(correct[-1] if k == 1 else 0 if slots <= 0 else 1 if equal <= slots else slots / equal)
    return dict(logs=logs, correct=correct, top=top, totals=[-sum(logs), len(labels) - sum(correct), len(labels) - sum(top)])


dropout = []
x = [[-2, 0, 3], [4, -5, 6]]
dy = [[1, -2, 3], [-4, 5, -6]]
for name, masks in [('all-keep', [[1]*3]*2), ('all-drop', [[0]*3]*2), ('mixed', [[1, 0, 1], [0, 1, 0]])]:
    for mode, enable in [('Train', True), ('Evaluate', True), ('Train', False), ('GradientCheck', True), ('FeatureGeneration', True), ('MultiviewTest', True)]:
        factors = masks if mode == 'Train' and enable else [[.5]*3]*2
        dropout.append(dict(name=f'{name}-{mode}-{enable}', features=3, cases=2, mode=mode, enable=enable, keep=.5,
                            inputs=flat(x), upstream=flat(dy), mask=flat(masks),
                            output=flat([[a*b for a, b in zip(row, f)] for row, f in zip(x, factors)]),
                            gradient=flat([[a*b for a, b in zip(row, f)] for row, f in zip(dy, factors)])))
# Multiplication, not a conditional: dropped NaN/inf remains NaN.
dropout.append(dict(name='masked-special', features=3, cases=1, mode='Train', enable=True, keep=.5,
                    inputs=[math.nan, math.inf, -math.inf], upstream=[math.inf, math.nan, -math.inf], mask=[0, 0, 1],
                    output=[math.nan, math.nan, -math.inf], gradient=[math.nan, math.nan, -math.inf]))

soft = []
for name, z in [('large', [[1000, 999, -1000], [-1000, -1001, -999]]),
                ('equal', [[0]*4, [-1000]*4]),
                ('negative-infinity', [[0, -math.inf, 1], [-math.inf]*3]),
                ('nan-first-middle', [[math.nan, 1, 2], [1, math.nan, 2]]),
                ('positive-infinity', [[math.inf, 0, -1]]),
                ('below-finite-max-seed', [[-3e38, -2.5e38]])]:
    p = [softmax(row) for row in z]
    u = [[(j+1)*.25*(-1 if (i+j)%2 else 1) for j in range(len(z[0]))] for i in range(len(z))]
    back = [[pc*(uc-sum(a*b for a, b in zip(ur, pr))) for pc, uc in zip(pr, ur)] for pr, ur in zip(p, u)]
    soft.append(dict(name=name, cases=len(z), classes=len(z[0]), logits=flat(z), upstream=flat(u), output=flat(p), gradient=flat(back)))

fused = []
for name, p, labels in [('regular', [[.125, .25, .625], [.5, .25, .25]], [0, 2]),
                        ('zero-probability', [[0, .5, .5]], [0])]:
    coeff = .75
    grad = [[coeff*((j == label)-v) for j, v in enumerate(row)] for row, label in zip(p, labels)]
    probgrad = [[coeff*(j == label)/v if v else math.inf if j == label else math.nan for j, v in enumerate(row)] for row, label in zip(p, labels)]
    fused.append(dict(name=name, cases=len(p), classes=len(p[0]), probabilities=flat(p), labels=labels, coefficient=coeff,
                      gradient=flat(grad), probability_gradient=flat(probgrad)))

costs = []
for name, p, labels, k in [
    ('ties-at-five', [[.25, .25]+[.0625]*6+[0, 0], [.25]*3+[.0625]*4+[0]*3, [.125]*8+[0]*2], [3, 0, 7], 5),
    ('top-one', [[.25]*4, [.5, .25, .125, .125]], [0, 3], 1),
    ('zeros', [[0, .5, .5], [0]*3], [0, 0], 5),
    ('nan-and-negative', [[.5, math.nan, .5], [-.125, .5, .625]], [1, 0], 5),
    ('class-0-999', [[1/1024]*999+[25/1024], [25/1024]+[1/1024]*999], [999, 0], 5),
]:
    costs.append(dict(name=name, cases=len(p), classes=len(p[0]), probabilities=flat(p), labels=labels, topk=k, **cost(p, labels, k)))

views = []
for count in (2, 10):
    ps = [[[.5, .25, .125, .125], [.125, .125, .25, .5]] if i%2 == 0 else
          [[.125, .125, .25, .5], [.5, .25, .125, .125]] for i in range(count)]
    avg = [[sum(ps[v][n][c] for v in range(count))/count for c in range(4)] for n in range(2)]
    views.append(dict(count=count, cases=2, classes=4, labels=[0, 3], topk=3, views=[flat(p) for p in ps], average=flat(avg), **cost(avg, [0, 3], 3)))

mask64 = (1 << 64) - 1

def uniform(seed, i):
    z = (seed + 0x9E3779B97F4A7C15*(i+1)) & mask64
    z = ((z ^ (z >> 30))*0xBF58476D1CE4E5B9) & mask64
    z = ((z ^ (z >> 27))*0x94D049BB133111EB) & mask64
    z ^= z >> 31
    return ((z >> 40)+1)/16777216

# Invert SplitMix's finalizer to pin both representable uniform endpoints.
def unxor(value, shift):
    result = value
    for _ in range(64 // shift + 1):
        result = value ^ (result >> shift)
    return result

endpoint_seed = unxor(((unxor(((unxor(mask64, 31) * pow(0x94D049BB133111EB, -1, 1 << 64)) & mask64), 27) * pow(0xBF58476D1CE4E5B9, -1, 1 << 64)) & mask64), 30)
endpoint_seed = (endpoint_seed - 0x9E3779B97F4A7C15) & mask64
rng = [dict(seed=str(seed), offset=str(offset), uniform=[uniform(seed, offset+i) for i in range(12)],
            mask=[int(uniform(seed, offset+i) < .5) for i in range(12)]) for seed, offset in [(0, 0), (10001, 12), (mask64, mask64-5), ((-0x9E3779B97F4A7C15) & mask64, 0), (endpoint_seed, 0)]]

def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return 'NaN' if math.isnan(x) else 'Infinity' if x > 0 else '-Infinity'
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x

# Separate float32 rounding witness for the original j-loop. Other oracles above
# use float64/exact arithmetic. This catches the tempting O(C) reassociation.
def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

wp = [.125, .25, .625]
wu = [-55865.36328125, 9107.64453125, 68786.7421875]
wg = []
for c in range(3):
    accum = 0.
    for j in range(3):
        accum = f32(accum + f32(wu[j] * f32((j == c) - wp[j])))
    wg.append(f32(accum * wp[c]))
witness = dict(probabilities=wp, upstream=wu, gradient=wg)

result = dict(rounding_witness=witness, oracle='Independent Python float64/exact; canonical feature-major/case-fastest arrays',
              dropout=dropout, softmax=soft, fused=fused, cost=costs, multiview=views, rng=rng)
Path(__file__).with_name('dropout-softmax-cost-cases.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False)+'\n')
