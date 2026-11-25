"""Offline, independent integer-arithmetic oracle for layers-145.shapes.json.
Consumes the ANX-2 golden parse, never calls the C# shape engine. Run from repo root.
Source: layer.py:370,649,705,880-884,999-1043,1067-1073,1099,1110,1145-1165,1195-1214,1263-1270.
"""
import json
from math import isqrt
from pathlib import Path

root = Path(__file__).resolve().parent
layers = json.loads((root / 'layers-145.parsed.json').read_text())['layers']
by_name = {l['name']: l for l in layers}


def infer(inner):
    computed = {}

    def visit(name):
        if name in computed:
            return computed[name]
        layer = by_name[name]
        p, kind, inputs = layer['parameters'], layer['type'], layer['inputs']
        ns = [visit(n)['outputs'] for n in inputs]
        s = dict(name=name, type=kind, inputs=inputs, num_inputs=ns, img_size=[], channels=[],
                 modules_x=None, modules=None, outputs_x=None, outputs=0, filter_channels=[],
                 filter_pixels=[], weights_shapes=[], biases_shape=[], partial_boundary_window=[])
        if kind == 'data':
            s['outputs'] = inner * inner * 3 if p['dataIdx'] == 0 else 1
        elif kind == 'conv':
            s['channels'] = p['channels']
            s['img_size'] = [isqrt(n // c) for n, c in zip(ns, p['channels'])]
            widths = [1 - (-(im - 2 * pad - f) // st) for im, pad, f, st in
                      zip(s['img_size'], p['padding'], p['filterSize'], p['stride'])]
            assert len(set(widths)) == 1
            s['modules_x'] = widths[0]
            s['modules'] = widths[0] ** 2
            filters = p['filters'][0] * p['groups'][0]
            s['outputs'] = s['modules'] * filters
            s['filter_channels'] = [c // g for c, g in zip(p['channels'], p['groups'])]
            s['filter_pixels'] = [f * f for f in p['filterSize']]
            s['weights_shapes'] = [[c * f, filters] for c, f in zip(s['filter_channels'], s['filter_pixels'])]
            s['biases_shape'] = [filters if p['sharedBiases'] else s['outputs'], 1]
            s['partial_boundary_window'] = [(w - 1) * st + f > im - 2 * pad for w, st, f, im, pad in
                                            zip(widths, p['stride'], p['filterSize'], s['img_size'], p['padding'])]
        elif kind in ('pool', 'cmrnorm', 'rnorm'):
            s['channels'] = [p['channels']]
            im = isqrt(ns[0] // p['channels'])
            s['img_size'] = [im]
            if kind == 'pool':
                w = p['outputsX'] or (1 - (-(im - p['start'] - p['sizeX']) // p['stride']))
                s['outputs_x'] = w
                s['outputs'] = w * w * p['channels']
                s['partial_boundary_window'] = [w > 0 and p['start'] + (w - 1) * p['stride'] + p['sizeX'] > im]
            else:
                s['outputs'] = (ns[0] // p['channels']) * p['channels']
        elif kind == 'fc':
            s['outputs'] = p['outputs']
            s['weights_shapes'] = [[n, p['outputs']] for n in ns]
            s['biases_shape'] = [1, p['outputs']]
        elif kind != 'cost.logreg':
            s['outputs'] = ns[0]
        computed[name] = s
        return s

    shapes = [visit(l['name']) for l in layers]
    return dict(inner_size=inner, classes=1000, layers=shapes,
                weight_parameter_count=sum(r * c for s in shapes for r, c in s['weights_shapes']),
                bias_parameter_count=sum(s['biases_shape'][0] * s['biases_shape'][1] for s in shapes if s['biases_shape']))


if __name__ == '__main__':
    fixture = dict(schema_version=1, cases=[infer(224), infer(227)])
    (root / 'layers-145.shapes.json').write_text(json.dumps(fixture, indent=2) + '\n')
    for case in fixture['cases']:
        print(case['inner_size'], case['weight_parameter_count'], case['bias_parameter_count'])
