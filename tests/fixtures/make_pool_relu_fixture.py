#!/usr/bin/env python3
"""Independent membership-set/scatter oracle. Standard library only; no port imports.
Binary32 rounding is explicit. JSON strings preserve NaN, infinities and -0 bits.
Max uses the candidate's finite seed and numeric fmax; average uses clipped area.
"""
import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def encode(x):
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return 'Infinity' if x > 0 else '-Infinity'
    if x == 0 and math.copysign(1, x) < 0:
        return '-0'
    return x


def fixture(name, size, channels=1, cases=1, start=0, outputs=0, values=None, pool='max', nan_grad=False):
    width = outputs or 1 - (-(size-start-3)//2)
    values = values if values is not None else [f32((r*7+n*3)%29-20) for r in range(channels*size*size) for n in range(cases)]
    upstream = [float((r*3+n)%7-3) for r in range(channels*width*width) for n in range(cases)]
    if nan_grad:
        upstream[0] = math.nan
    result, dx = [], [0.0]*len(values)
    # Construct each window as a membership set, independently of input-centric undo.
    for c in range(channels):
        for oy in range(width):
            for ox in range(width):
                pixels = [y*size+x for y in range(size) for x in range(size)
                          if start+2*oy <= y < start+2*oy+3 and start+2*ox <= x < start+2*ox+3]
                for n in range(cases):
                    vals = [values[(c*size*size+p)*cases+n] for p in pixels]
                    if pool == 'max':
                        numeric = [f32(-2e38)] + [v for v in vals if not math.isnan(v)]
                        maximum = max(numeric)
                        if maximum == 0:
                            maximum = -0.0 if all(math.copysign(1,v)<0 for v in numeric if v == 0) else 0.0
                        value = maximum
                    else:
                        value = 0.0
                        for v in vals:
                            value = f32(value+v)
                        value = f32(value/len(vals)) if vals else math.nan
                    oi = ((c*width+oy)*width+ox)*cases+n
                    result.append(value)
                    for p in pixels:
                        i = (c*size*size+p)*cases+n
                        factor = float(values[i] == value) if pool == 'max' else f32(1/len(vals))
                        dx[i] = f32(dx[i]+f32(factor*upstream[oi]))
    return dict(name=name, size=size, channels=channels, cases=cases, start=start, outputs_x=width,
                pool=pool, input=list(map(encode, values)), dy=list(map(encode, upstream)),
                forward=list(map(encode, result)), dx=list(map(encode, dx)))


cases = [fixture('full-5',5),fixture('partial-6',6),fixture('full-7',7),
         fixture('explicit-full-6',6,outputs=2),fixture('partial-5-start1',5,start=1),
         fixture('partial-7-start1',7,start=1),fixture('left-top-clipped',5,start=-1),
         fixture('multiple-channels-cases',6,channels=2,cases=3),
         fixture('negative-only',5,values=[-float(i+1) for i in range(25)])]
ties = [-5.0]*25
for i in [1,2,7,11,12,13]:
    ties[i] = 9.0
cases += [fixture('internal-and-shared-ties',5,values=ties),fixture('signed-zeros',5,values=[-0.0,0.0]*12+[-0.0]),
          fixture('negative-zero-only',5,values=[-0.0]*25),
          fixture('mixed-nan',5,values=[math.nan if i%3==0 else -float(i) for i in range(25)]),
          fixture('all-nan',5,values=[math.nan]*25),
          fixture('below-finite-seed',5,values=[-math.inf if i%2 else f32(-3e38) for i in range(25)]),
          fixture('nan-upstream-false-mask',5,values=[float(i) for i in range(25)],nan_grad=True),
          fixture('empty-final-window',5,outputs=4)]
for size in [5,6,7]:
    cases.append(fixture('average-'+str(size),size,channels=2,cases=3,pool='avg'))
cases.append(fixture('average-clipped-start',5,start=-1,pool='avg'))
x = [-2., -0., 0., 3., math.nan, -math.inf, math.inf]
dy = [-3., -4., 2., 5., 7., math.nan, -2.]
y = [0. if v<0 else v for v in x]
dx = [f32(g*float(v>0)) for g,v in zip(dy,y)]
data = dict(pool=cases, relu={k:list(map(encode,v)) for k,v in dict(input=x,dy=dy,forward=y,dx=dx).items()})
assert [c['outputs_x'] for c in cases[:3]] == [2,3,3]
Path(__file__).with_name('pool-relu-cases.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
print('Generated',len(cases),'pool fixtures and one ReLU special-value fixture')
