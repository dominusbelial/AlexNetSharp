"""Generate content from the committed parser/shape fixtures; no candidate source."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
layers = json.loads((root / 'tests/fixtures/layers-145.parsed.json').read_text())['layers']
shapes = {s['name']: s for s in json.loads((root / 'tests/fixtures/layers-145.shapes.json').read_text())['cases'][0]['layers']}
# Summary, explanation, significance, equation, paper section, original citations.
types = {
 'data': ('The image or class index entering the graph.', 'Image data is planar RGB. The labels stream is a class index. This replay uses a synthetic image and label 0.', 'Both towers see the same crop. A synthetic label is only a cost demonstration.', 'x[c,y,x] = RGB[c,y,x] - mean[c]; label = 0', '2 / Dataset', 'layer.py:1099; convdata_jpeg.py:73'),
 'conv': ('Shared filters scan local windows to build feature maps.', 'Each output channel uses its own small RGB or feature-space filter at every location. Multiple inputs keep independent weight matrices in input order.', 'Ceiling arithmetic preserves a partial final window. Conv1 has no immediate ReLU; signed responses reach normalization.', 'y[f,y,x] = bias[f] + sum_inputs sum_c,ky,kx W[c,ky,kx,f] * input[c,y*stride+start+ky,x*stride+start+kx]; outside input = 0', '3.5 / Overall architecture', 'layer.py:1004; layer.py:1041; src/layer.cu:829'),
 'pool': ('Keep the strongest response in each local window.', 'Max pooling takes the largest real input in a clipped window, separately for each channel. In this configuration the windows overlap.', 'It reduces spatial size while allowing small position changes. Pool1 and pool3 have separate following ReLUs; pool2 does not.', 'y[c,y,x] = max(input[c, clipped(start+y*stride .. +sizeX-1), clipped(start+x*stride .. +sizeX-1)]); seed = -2e38', '3.4 / Overlapping pooling', 'layer.py:1164; src/layer.cu:1327'),
 'cmrnorm': ('Nearby channels compete at the same image position.', 'Cross-map response normalization divides each response by a power of the squared responses in a channel window. Its configured base is 2.', 'This operates on signed conv1 outputs before pooling and ReLU. Negative responses still contribute their squares.', 'denom = minDiv + (rawScale/size) * sum_channel_window x²; y = x * denom^(-pow)', '3.3 / Local response normalization', 'layer.py:1183; src/layer.cu:1589'),
 'rnorm': ('Normalize a spatial neighbourhood within one channel.', 'Spatial response normalization gathers a square neighbourhood in the same channel. It is a different operation from cross-map normalization.', 'The released 145 configuration adds this stage after conv2 normalization although the paper does not describe it.', 'denom = 1 + (rawScale/size²) * sum_spatial_window x²; y = x * denom^(-pow)', '3.3 / Code-specific extra stage', 'layer.py:1183; src/layer.cu:1569'),
 'neuron': ('A separate ReLU removes negative responses.', 'The original parser detaches neuron=relu into an explicit layer and redirects downstream inputs to it. ReLU keeps positive values and maps negative values to zero.', 'Showing this node preserves the exact order of convolution, normalization, pooling and activation.', 'y = x < 0 ? 0 : x', '3.1 / ReLU nonlinearity', 'layer.py:627; include/neuron.cuh:163'),
 'fc': ('Combine every input feature with learned weighted sums.', 'Fully connected layers keep one weight matrix per input. The bias is added after the first product, before later products. The four 2048 nodes have separate following ReLUs.', 'Cross-tower fan-in joins both halves. fc2048bb intentionally reads hs1b before hs1a; its first weight matrix belongs to hs1b.', 'y = (X[0] W[0] + bias) + X[1] W[1]; reduction follows declared input order', '3.5 / Fully connected layers', 'src/layer.cu:640; src/layer.cu:644; layers/layers-145.cfg:277'),
 'hs': ('Evaluation multiplies the response by the keep probability.', 'Training would use an unscaled Bernoulli mask. This replay is evaluation mode and makes no random dropout draws.', 'The original convention uses keep multiplication during evaluation, so omitting this node would change every later response.', 'y = keep * x (evaluation); y = Bernoulli(keep) * x (training)', '4.2 / Dropout', 'src/layer.cu:1127; src/layer.cu:1133'),
 'softmax': ('Turn the 1000 logits into class probabilities.', 'Subtract the largest logit, exponentiate, and divide by the total. Class labels are shown only as indices because this slice has no ImageNet class map.', 'Untrained probabilities have no predictive meaning. Subtracting the maximum avoids overflow for finite logits.', 'p[c] = exp(z[c]-max(z)) / sum_j exp(z[j]-max(z))', '3 / 1000-way softmax', 'src/layer.cu:1020; src/layer.cu:1026'),
 'cost.logreg': ('Measure the synthetic label probability and ranking errors.', 'The cost reads labels and probabilities. It reports negative log probability, top-1 error and top-5 error; ties receive fractional credit.', 'This is an evaluation measurement for one artificial label, not evidence of classification quality or training.', 'NLL = -log(p[label]); top-k error = 1 - clamp((k-B)/E,0,1), B = larger probabilities, E = equal probabilities', '3 / Multinomial logistic regression', 'src/layer.cu:1825; src/layer_kernels.cu:106'),
}

def write(id, title, summary, section, citations, explanation, why, formula, links):
    text = f'---\nid: {id}\ntitle: {title}\nsummary: {summary}\npaper-section: {section}\ncitations: {citations}\n---\n\n{explanation}\n\nWhy it matters: {why}\n\n' + ' · '.join(f'[{link}](card:{link})' for link in links) + f'\n\nFormula: {formula}\n'
    (root / 'content/explain' / (id + '.md')).write_text(text)

for type, (summary, explanation, why, equation, section, citations) in types.items():
    write('type-' + type, type, summary, section, citations, explanation, why, equation,
          [l['name'] for l in layers if l['type'] == type])

cfg_lines = (root / 'layers/layers-145.cfg').read_text().splitlines()
for layer in layers:
    name, type = layer['name'], layer['type']
    summary, explanation, why, equation, section, citations = types[type]
    shape = shapes[name]
    source = layer['sourceLayer'] or name
    line = cfg_lines.index('[' + source + ']') + 1
    params = '; '.join(f'{key}={value}' for key, value in layer['parameters'].items())
    outputs = shape['outputs']
    counts = sum(w[0] * w[1] for w in shape['weights_shapes']) + (shape['biases_shape'][0] * shape['biases_shape'][1] if shape['biases_shape'] else 0)
    extra = f'This instance has {outputs:,} output values per image and {counts:,} parameter slots. Effective inputs in order: ' + (', '.join(layer['inputs']) or 'none') + f'.\n\nConfiguration: {params}.\n\nLive shape, activation statistics and parameter values are supplied by the loaded snapshot.'
    if layer['sourceLayer']:
        extra += ' This is the explicit ReLU after ' + source + '.'
    if name.startswith('pool3'):
        extra += ' This is the pool after conv5, often called pool5; the config names it pool3.'
    links = ['type-' + type] + layer['inputs'] + [l['name'] for l in layers if name in l['inputs']]
    write(name, name, summary, section, f'layers/layers-145.cfg:{line}; {citations}', extra, why, equation, links)
print(f'Wrote {len(layers) + len(types)} explain cards.')
