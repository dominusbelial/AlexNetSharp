#!/usr/bin/env python3
"""Independent standard-library float64 oracle: enumerate membership output-first,
then scatter its Jacobian contributions. No C# calls or input-centric undo bounds.
Nonfinite values are strings in strict JSON. Runtime float32 comparisons use the
stated tolerance; pow and ordered float32 reductions cannot be exact float64.
"""
import json
import math
from pathlib import Path


def fixture(name, kind, channels, width, size=5, raw_scale=0.0001,
            minimum=2.0, cases=3, blocked=False, values=None, gradients=None):
    count = channels * width * width * cases
    x = values or [((i * 13 + 7) % 31 - 15) / 4 for i in range(count)]
    dy = gradients or [((i * 7) % 19 - 9) / 8 for i in range(count)]
    assert len(x) == len(dy) == count
    scale = raw_scale / (size if kind == 'cmrnorm' else size ** 2)
    coords = [(c, y, z) for c in range(channels) for y in range(width) for z in range(width)]
    denom, output, dx = [0.] * count, [0.] * count, [0.] * count
    windows = []
    for c, y, z in coords:
        start = c // size * size if blocked else c - size // 2
        windows.append([j for j, (cc, yy, zz) in enumerate(coords)
                        if (yy == y and zz == z and start <= cc < start + size)
                        if kind == 'cmrnorm'] if kind == 'cmrnorm' else
                       [j for j, (cc, yy, zz) in enumerate(coords)
                        if cc == c and y - size // 2 <= yy < y - size // 2 + size
                        and z - size // 2 <= zz < z - size // 2 + size])
    for r, window in enumerate(windows):
        for n in range(cases):
            i = r * cases + n
            denom[i] = (minimum if kind == 'cmrnorm' else 1.) + scale * sum(x[j * cases + n] ** 2 for j in window)
            output[i] = x[i] * denom[i] ** -0.75
    # Output-centric scatter is deliberately different from the production gather.
    for r, window in enumerate(windows):
        for n in range(cases):
            i = r * cases + n
            dx[i] += dy[i] * denom[i] ** -0.75
            factor = -2 * scale * 0.75 * dy[i] * output[i] / denom[i]
            for j in window:
                dx[j * cases + n] += factor * x[j * cases + n]
    return dict(name=name, kind=kind, channels=channels, width=width, size=size,
                cases=cases, raw_scale=raw_scale, scale=scale, power=0.75,
                minimum=minimum, blocked=blocked, input=x, upstream=dy,
                denominators=denom, output=output, backward=dx)


fixtures = []
for channels in (3, 5, 7):
    fixtures.append(fixture(f'cross-clamp-C{channels}', 'cmrnorm', channels, 1))
fixtures += [fixture('cross-minDiv1', 'cmrnorm', 7, 1, minimum=1),
             fixture('cross-multiple-pixels', 'cmrnorm', 7, 2),
             fixture('cross-even', 'cmrnorm', 7, 2, size=4, raw_scale=0.2),
             fixture('cross-blocked-partial', 'cmrnorm', 7, 2, size=3, blocked=True, raw_scale=0.2)]
for width in (1, 2, 6):
    fixtures.append(fixture(f'spatial-clamp-{width}', 'rnorm', 2, width, raw_scale=0.001))
fixtures += [fixture('spatial-even', 'rnorm', 2, 6, size=4, raw_scale=0.2),
             fixture('spatial-ignores-minDiv-blocked', 'rnorm', 2, 2, minimum=99, blocked=True, raw_scale=0.001)]
for kind in ('cmrnorm', 'rnorm'):
    channels, width = (7, 1) if kind == 'cmrnorm' else (1, 6)
    count = channels * width * width * 3
    fixtures.append(fixture(kind+'-signed-zero', kind, channels, width,
                            values=[-0. if i % 2 else 0. for i in range(count)]))
    fixtures.append(fixture(kind+'-large', kind, channels, width,
                            values=[(-1 if i % 2 else 1) * (i + 1) * 1e8 for i in range(count)]))
    for label, special, upstream in [('NaN', math.nan, False), ('inf', math.inf, False),
                                      ('negative-inf', -math.inf, False), ('NaN-upstream', math.nan, True)]:
        values = [float(i % 7 - 3) for i in range(count)]
        grads = [float(i % 5 - 2) for i in range(count)]
        (grads if upstream else values)[count // 2] = special
        fixtures.append(fixture(kind+'-'+label, kind, channels, width, values=values, gradients=grads))


def encode(v):
    if isinstance(v, float):
        if math.isnan(v): return 'NaN'
        if math.isinf(v): return 'Infinity' if v > 0 else '-Infinity'
        if v == 0 and math.copysign(1, v) < 0: return '-0'
    if isinstance(v, list): return [encode(x) for x in v]
    if isinstance(v, dict): return {k: encode(x) for k, x in v.items()}
    return v


path = Path(__file__).with_name('norm-cases.json')
path.write_text(json.dumps(encode(dict(oracle='float64 output-centric Jacobian scatter',
                                      tolerance='4e-6 * (1 + abs(expected))', fixtures=fixtures)), indent=2, allow_nan=False) + '\n')
print(f'Wrote {len(fixtures)} independent norm fixtures')
