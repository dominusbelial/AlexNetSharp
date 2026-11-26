"""Independent float64 FC oracle, standard library only. Binary-fraction tiny inputs
keep finite fixtures exact in float32; sums enumerate cases/outputs, not SIMD lanes.
Physical arrays are column-major; nonfinite JSON values are explicit strings.
"""
import json
import math
from pathlib import Path


def encode(v):
    if isinstance(v, list):
        return [encode(x) for x in v]
    if isinstance(v, dict):
        return {k: encode(x) for k, x in v.items()}
    if isinstance(v, float):
        if math.isnan(v): return 'NaN'
        if math.isinf(v): return 'Infinity' if v > 0 else '-Infinity'
        if v == 0 and math.copysign(1, v) < 0: return '-0'
    return v


def fixture(name, widths, cases, outputs, alpha=.375, beta=-.25, special=False, reverse=False):
    xs = [[((j * 3 + i * 5) % 13 - 6) / 4 for j in range(cases * k)] for i, k in enumerate(widths)]
    ws = [[((j * 5 + i * 7) % 17 - 8) / 8 for j in range(k * outputs)] for i, k in enumerate(widths)]
    b = [(o - 2) / 2 for o in range(outputs)]
    dy = [((j * 7) % 11 - 5) / 4 for j in range(cases * outputs)]
    names = ['hs1b', 'hs1a'] if reverse else ['input' + str(i) for i in range(len(widths))]
    if reverse: xs.reverse()  # W slots stay in declared order, as in fc2048bb.
    if special:
        xs[0][:5] = [0., -0., -2., float('nan'), float('inf')]
        dy[:5] = [-0., 0., float('-inf'), float('nan'), 2.]
    y = []
    for o in range(outputs):
        for n in range(cases):
            total = sum(xs[0][k * cases + n] * ws[0][o * widths[0] + k] for k in range(widths[0])) + b[o]
            for i in range(1, len(widths)):
                total += sum(xs[i][k * cases + n] * ws[i][o * widths[i] + k] for k in range(widths[i]))
            y.append(total)
    dxs, dws = [], []
    for i, width in enumerate(widths):
        dxs.append([sum(dy[o * cases + n] * ws[i][o * width + k] for o in range(outputs)) for k in range(width) for n in range(cases)])
        dws.append([beta * 3 + (0. if alpha == 0 else alpha * sum(xs[i][k * cases + n] * dy[o * cases + n] for n in range(cases))) for o in range(outputs) for k in range(width)])
    db = [beta * 3 + alpha * sum(dy[o * cases + n] for n in range(cases)) for o in range(outputs)]
    return dict(name=name, widths=widths, cases=cases, outputs=outputs, input_names=names,
                inputs=xs, weights=ws, bias=b, upstream=dy, alpha=alpha, beta=beta,
                output=y, input_gradients=dxs, weight_gradients=dws, bias_gradient=db)


fixtures = [fixture('one-input-N1', [3], 1, 5), fixture('two-widths-N2', [3, 5], 2, 9),
            fixture('two-widths-N3', [9, 11], 3, 11), fixture('two-widths-N4', [5, 3], 4, 3),
            fixture('two-widths-N5', [3, 5], 5, 9), fixture('hs-order-ab', [3, 3], 3, 5),
            fixture('hs-order-ba', [3, 3], 3, 5, reverse=True),
            fixture('nonfinite', [3, 5], 5, 9, special=True),
            fixture('zero-alpha', [3, 5], 5, 9, alpha=0, beta=.5, special=True)]
# Shared width=2, two outputs, two consumers; each product independently overwrites.
up_a, up_b = [2., -3.], [4., 1.]
w_a, w_b = [1., 2., 3., 4.], [5., 6., 7., 8.]
a = [sum(up_a[o] * w_a[o * 2 + k] for o in range(2)) for k in range(2)]
b = [sum(up_b[o] * w_b[o * 2 + k] for o in range(2)) for k in range(2)]
result = dict(fixtures=fixtures, fanout=dict(upstream_a=up_a, upstream_b=up_b, weights_a=w_a, weights_b=w_b,
                                            local_a=a, local_b=b, merged=[x+y for x, y in zip(a,b)]))
Path(__file__).with_name('fc-cases.json').write_text(json.dumps(encode(result), indent=2, allow_nan=False) + '\n')
