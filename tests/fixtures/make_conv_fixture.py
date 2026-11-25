#!/usr/bin/env python3
"""Independent exact-rational oracle: explicitly assemble the linear map from input
pixels to output pixels, then apply that map and its transpose. No port invocation,
third-party library or CUDA source. Run from any directory; rewrites conv-cases.json.

5/3/2 and 15/11/4 tile exactly: the brief's requested sizes do NOT have a
partial window. Retain those exact sizes and add 6/3/2 and 16/11/4 as edge proofs.
"""
from fractions import Fraction as Q
from pathlib import Path
import json


def values(count, phase):
    return [Q(((i * 7 + phase) % 13) - 6, 2) for i in range(count)]


def fixture(name, geometries, filters=1, cases=1):
    # Geometry tuples: image side, channels, filter side, stride, config padding.
    widths = [1 + -(-(s + 2*p - k) // t) for s,c,k,t,p in geometries]
    assert len(set(widths)) == 1
    out = widths[0]
    rows = filters * out * out
    dy = values(rows * cases, 5)
    bias = values(filters, 9)
    y = [bias[r // (out*out)] for r in range(rows) for n in range(cases)]
    db = [sum((dy[(f*out*out+m)*cases+n] for m in range(out*out) for n in range(cases)), Q()) for f in range(filters)]
    inputs = []
    for s,c,k,t,p in geometries:
        x = values(c*s*s*cases, 2 + len(inputs))
        w = values(c*k*k*filters, 4 + len(inputs))
        dx = [Q()] * len(x)
        dw = [Q()] * len(w)
        # Assemble sparse entries (output row, input feature, weight row).
        # Input-centric membership gives an independent traversal from the port.
        entries = []
        for feature in range(c*s*s):
            ch, spatial = divmod(feature, s*s)
            iy, ix = divmod(spatial, s)
            for row in range(rows):
                f, spatial_out = divmod(row, out*out)
                oy, ox = divmod(spatial_out, out)
                ky, kx = iy + p - oy*t, ix + p - ox*t
                if 0 <= ky < k and 0 <= kx < k:
                    entries.append((row, feature, (ch*k+ky)*k+kx, f))
        for row, feature, wr, f in entries:
            wi = wr*filters+f
            for n in range(cases):
                xi, oi = feature*cases+n, row*cases+n
                y[oi] += x[xi]*w[wi]
                dx[xi] += dy[oi]*w[wi]
                dw[wi] += x[xi]*dy[oi]
        inputs.append(dict(size=s, channels=c, filter=k, stride=t, padding=p,
                           images=x, weights=w, dx=dx, dw=dw,
                           dw_training=[v/cases for v in dw]))
    return dict(name=name, filters=filters, cases=cases, modules_x=out,
                partial_sum=out, inputs=inputs, biases=bias, dy=dy, forward=y,
                db=db, db_training=[v/cases for v in db])


cases = [fixture('requested-5x5-full', [(5,1,3,2,0)]),
         fixture('requested-5x5-pad1', [(5,1,3,2,1)]),
         fixture('ceiling-6x6-partial', [(6,1,3,2,0)]),
         fixture('two-channels-filters-cases', [(4,2,3,2,0)], 2, 2),
         fixture('requested-15x15-k11-s4-full', [(15,1,11,4,0)]),
         fixture('mini-conv1-16x16-partial', [(16,1,11,4,0)]),
         fixture('two-inputs-distinct-geometry', [(6,1,3,2,0),(3,2,1,1,0)], 2, 2)]
Path(__file__).with_name('conv-cases.json').write_text(json.dumps(
    {'oracle': 'Exact rational sparse linear map and transpose; make_conv_fixture.py', 'cases':cases},
    default=float, indent=2) + '\n')
print(f'Generated {len(cases)} independent rational convolution fixtures.')
