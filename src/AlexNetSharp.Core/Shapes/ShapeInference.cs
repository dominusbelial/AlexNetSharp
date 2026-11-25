using AlexNetSharp.Core.Config;

namespace AlexNetSharp.Core.Shapes;

/// <summary>Shape arithmetic only; no allocation of tensors/weights and no layer maths.
/// Configuration-only checks remain in the ANX-2 parser.</summary>
public static class ShapeInference
{
    public static NetworkShapes Infer(NetworkDefinition network, DataDimensions? data = null)
    {
        data ??= DataDimensions.Jpeg();
        var done = new Dictionary<string, LayerShape>(StringComparer.Ordinal);
        var visiting = new HashSet<string>(StringComparer.Ordinal);
        var hasWeightsBelow = new Dictionary<string, bool>(StringComparer.Ordinal);

        bool GradConsumersBelow(LayerDefinition layer)
        {
            if (hasWeightsBelow.TryGetValue(layer.Name, out bool result)) return result;
            // During parse(), weighted layers have gradConsumer=True (layer.py:830).
            // Rates overwrite it only in add_params(), AFTER all shape checks (:293,314,744).
            return hasWeightsBelow[layer.Name] = layer.Kind is LayerKind.Conv or LayerKind.Fc ||
                layer.Inputs.Any(n => GradConsumersBelow(network[n]));
        }

        LayerShape Visit(LayerDefinition layer)
        {
            if (done.TryGetValue(layer.Name, out var known)) return known;
            if (!visiting.Add(layer.Name)) throw new ArgumentException($"Cyclic network at '{layer.Name}'.", nameof(network));
            var inputs = layer.Inputs.Select(n => Visit(network[n])).ToArray();
            foreach (var input in inputs)
                if (input.Outputs == 0) Fail(layer, $"Layer '{layer.Name}': input layer '{input.Name}' does not produce any output");

            var numInputs = inputs.Select(s => s.Outputs).ToArray();
            int[] imgSize = [], channels = [], filterChannels = [], filterPixels = [], biases = [];
            int[][] weights = [];
            bool[] partial = [];
            int? modulesX = null, modules = null, outputsX = null;
            int outputs;
            switch (layer.TypedParams)
            {
                case DataLayerParams p:
                    if (!data.Dimensions.TryGetValue(p.DataIdx, out outputs))
                        throw new ArgumentException($"No provider dimension for dataIdx={p.DataIdx}.", nameof(data));
                    break;
                case ConvLayerParams p:
                {
                    channels = p.Channels.ToArray();
                    var pixels = numInputs.Select((n, i) => n / channels[i]).ToArray(); // Python 2 integer division
                    imgSize = pixels.Select(n => (int)Math.Sqrt(n)).ToArray();
                    for (int i = 0; i < imgSize.Length; i++)
                        NetworkConfigValidators.VerifyIntRange(layer.Name, imgSize[i], "imgSize", 1, null, i);
                    var widths = imgSize.Select((s, i) => 1 + Ceil(s - 2 * p.PaddingStart[i] - p.FilterSize[i], p.Stride[i])).ToArray();
                    if (widths.Distinct().Count() != 1 || p.Filters.Select((f, i) => f * p.Groups[i]).Distinct().Count() != 1)
                        Fail(layer, $"Layer '{layer.Name}': all inputs must produce equally-dimensioned output. Dimensions are: {string.Join(", ", widths.Select((w, i) => $"{p.Filters[i] * p.Groups[i]}x{w}x{w}"))}.");
                    modulesX = widths[0];
                    modules = checked(modulesX.Value * modulesX.Value);
                    outputs = checked(modules.Value * p.OutputFilters);
                    filterChannels = (p.ExplicitFilterChannels ?? channels.Select((c, i) => c / p.Groups[i]).ToArray()).ToArray();
                    filterPixels = p.FilterSize.Select(f => checked(f * f)).ToArray();
                    partial = new bool[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        if (numInputs[i] % pixels[i] != 0 || imgSize[i] * imgSize[i] != pixels[i])
                            Fail(layer, $"Layer '{layer.Name}[{i}]': has {numInputs[i]} dimensional input, not interpretable as square {channels[i]}-channel images");
                        int paddedSize = imgSize[i] - 2 * p.PaddingStart[i];
                        if (p.FilterSize[i] > paddedSize)
                            Fail(layer, $"Layer '{layer.Name}[{i}]': filter size ({p.FilterSize[i]}) greater than image size + 2 * padding ({paddedSize})");
                        partial[i] = (modulesX.Value - 1) * p.Stride[i] + p.FilterSize[i] > paddedSize;
                    }
                    if (p.PartialSum != 0 && modules.Value % p.PartialSum != 0)
                        Fail(layer, $"Layer '{layer.Name}': convolutional layer produces {modulesX}x{modulesX}={modules} outputs per filter, but given partialSum parameter ({p.PartialSum}) does not divide this number");
                    weights = filterChannels.Select((c, i) => new[] { checked(c * filterPixels[i]), p.OutputFilters }).ToArray();
                    biases = [p.SharedBiases ? p.OutputFilters : outputs, 1];
                    break;
                }
                case PoolLayerParams p:
                {
                    channels = [p.Channels];
                    int pixels = numInputs[0] / p.Channels;
                    int size = (int)Math.Sqrt(pixels);
                    imgSize = [size];
                    NetworkConfigValidators.VerifyIntRange(layer.Name, p.SizeX, "sizeX", 1, size);
                    NetworkConfigValidators.VerifyIntRange(layer.Name, p.Stride, "stride", 1, p.SizeX);
                    if (GradConsumersBelow(layer))
                        NetworkConfigValidators.VerifyDivisible(layer.Name, true, p.Channels, 16, "channels");
                    VerifyImage(layer, numInputs[0], pixels, size, p.Channels);
                    outputsX = p.OutputsX <= 0 ? Ceil(size - p.Start - p.SizeX, p.Stride) + 1 : p.OutputsX;
                    outputs = checked(outputsX.Value * outputsX.Value * p.Channels);
                    partial = [outputsX > 0 && p.Start + (outputsX.Value - 1) * p.Stride + p.SizeX > size];
                    break;
                }
                case CmrNormLayerParams p:
                    (imgSize, channels, outputs) = Norm(layer, numInputs[0], p.Channels, p.Size, true);
                    break;
                case RNormLayerParams p:
                    (imgSize, channels, outputs) = Norm(layer, numInputs[0], p.Channels, p.Size, false);
                    break;
                case FcLayerParams p:
                    outputs = p.Outputs;
                    weights = numInputs.Select(n => new[] { n, outputs }).ToArray();
                    biases = [1, outputs];
                    break;
                case LogregCostLayerParams:
                    if (numInputs[0] != 1) Fail(layer, $"Layer '{layer.Name}': dimensionality of first input must be 1");
                    if (network[layer.Inputs[1]].Kind != LayerKind.Softmax) Fail(layer, $"Layer '{layer.Name}': second input must be softmax layer");
                    if (numInputs[1] != data.Classes)
                        Fail(layer, $"Layer '{layer.Name}': softmax input '{layer.Inputs[1]}' must produce {data.Classes} outputs, because that is the number of classes in the dataset");
                    outputs = 0; // LayerParser.set_defaults, layer.py:187; cost has no activation output
                    break;
                case NeuronLayerParams or DropoutLayerParams or SoftmaxLayerParams:
                    outputs = numInputs[0]; // layer.py:649,705,1110
                    break;
                default:
                    throw new NotSupportedException($"Shape inference for '{layer.Type}' is unsupported.");
            }
            var shape = new LayerShape(layer.Name, layer.Type, layer.Inputs.ToArray(), numInputs,
                imgSize, channels, modulesX, modules, outputsX, outputs, filterChannels, filterPixels, weights, biases, partial);
            VerifySharedWeights(layer, shape, network, Visit);
            visiting.Remove(layer.Name);
            return done[layer.Name] = shape;
        }

        return new NetworkShapes(network.Layers.Select(Visit).ToArray());
    }

    private static int Ceil(int numerator, int denominator) => (int)Math.Ceiling(numerator / (double)denominator);

    private static (int[], int[], int) Norm(LayerDefinition layer, int input, int channels, int window, bool crossMap)
    {
        int pixels = input / channels;
        int size = (int)Math.Sqrt(pixels);
        if (!crossMap) NetworkConfigValidators.VerifyIntRange(layer.Name, window, "size", 1, size);
        VerifyImage(layer, input, pixels, size, channels);
        return ([size], [channels], checked(pixels * channels));
    }

    private static void VerifyImage(LayerDefinition layer, int input, int pixels, int size, int channels)
    {
        // Preserve original integer-division/modulo predicate, including its small-input quirks.
        if (pixels == 0 || input % pixels != 0 || size * size != pixels)
            Fail(layer, $"Layer '{layer.Name}': has {input} dimensional input, not interpretable as {channels}-channel images");
    }

    private static void VerifySharedWeights(LayerDefinition layer, LayerShape shape, NetworkDefinition network, Func<LayerDefinition, LayerShape> visit)
    {
        var sources = layer.TypedParams switch
        {
            ConvLayerParams p => p.WeightSource,
            FcLayerParams p => p.WeightSource,
            _ => Array.Empty<string>(),
        };
        for (int i = 0; i < sources.Count; i++)
        {
            if (sources[i].Length == 0) continue;
            var source = NetworkConfigValidators.TryParseWeightSource(sources[i])!.Value;
            var matrix = (source.LayerName == layer.Name ? shape : visit(network[source.LayerName])).WeightsShapes[source.MatrixIdx];
            var expected = shape.WeightsShapes[i];
            if (!matrix.SequenceEqual(expected))
                Fail(layer, $"Layer '{layer.Name}': weight sharing source matrix '{sources[i]}' has shape {matrix[0]}x{matrix[1]}; should be {expected[0]}x{expected[1]}.");
        }
    }

    private static void Fail(LayerDefinition layer, string message) => throw new LayerParsingException(message, layer.Name);
}
