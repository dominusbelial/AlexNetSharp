using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Conv;

/// <summary>Square image/filter geometry. PaddingStart is already negated by the parser.
/// Module count uses ceiling, including a zero-padded final partial window.</summary>
public sealed record ConvInputGeometry(int ImageSize, int Channels, int FilterSize, int Stride,
    int PaddingStart = 0, int Groups = 1, bool RandSparse = false)
{
    public int ModulesX => checked(1 + (int)Math.Ceiling(
        (ImageSize - 2.0 * PaddingStart - FilterSize) / Stride));
}

/// <summary>Scalar float32 reference; no SIMD, im2col or implicit layout conversions.
/// Images/outputs are ANX-4 canonical (channel*y*x,case), weights (channel*fy*fx,filter).
/// Forward reduces channel,fy,fx; input gradients reduce my,mx,filter; weights reduce
/// consecutive module chunks, then module,case, scale each chunk, then add chunks in order.
/// Bias reduces module,case. These orders define CPU results, not historical CUDA bit parity.
/// Original src/layer.cu:823-895; A:src/cudaconv2/filter_acts.cu:129-154,
/// img_acts.cu:92-113, weight_acts.cu:76,112-115,186-196,599-601.</summary>
public sealed class ScalarConvolution
{
    public string Name { get; }
    public IReadOnlyList<ConvInputGeometry> Inputs { get; }
    public int Filters { get; }
    public int ModulesX { get; }
    public int Modules => checked(ModulesX * ModulesX);
    public int PartialSum { get; }
    public bool SharedBiases { get; }

    public ScalarConvolution(string name, IReadOnlyList<ConvInputGeometry> inputs, int filters,
        int partialSum = 0, bool sharedBiases = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Count == 0) throw new ArgumentException("Convolution needs an input.", nameof(inputs));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(filters);
        Name = name;
        foreach (var g in inputs)
        {
            if (g.Groups != 1 || g.RandSparse)
                throw new NotSupportedException($"Layer '{name}': grouped or sparse convolution is unsupported in this port (groups={g.Groups}, randSparse={g.RandSparse}).");
            if (g.ImageSize <= 0 || g.Channels <= 0 || g.FilterSize <= 0 || g.Stride <= 0 ||
                g.PaddingStart > 0 || g.Stride > g.FilterSize || g.ModulesX <= 0)
                throw new ArgumentException($"Layer '{name}': invalid convolution geometry.", nameof(inputs));
            _ = checked(g.Channels * g.ImageSize * g.ImageSize);
            _ = checked(g.Channels * g.FilterSize * g.FilterSize * filters);
        }
        Inputs = Array.AsReadOnly(inputs.ToArray());
        ModulesX = inputs[0].ModulesX;
        if (inputs.Any(g => g.ModulesX != ModulesX))
            throw new ArgumentException($"Layer '{name}': inputs must have equal module widths.", nameof(inputs));
        if (partialSum < 0 || (partialSum > 0 && Modules % partialSum != 0))
            throw new ArgumentException($"Layer '{name}': partialSum must be zero or divide {Modules}.", nameof(partialSum));
        _ = checked(filters * Modules);
        Filters = filters;
        PartialSum = partialSum;
        SharedBiases = sharedBiases;
    }

    public static ScalarConvolution FromLayer(LayerDefinition layer, LayerShape shape)
    {
        if (layer.TypedParams is not ConvLayerParams p || shape.Type != "conv" || layer.Name != shape.Name)
            throw new ArgumentException("Matching convolution definition and shape required.");
        var inputs = Enumerable.Range(0, layer.Inputs.Count).Select(i => new ConvInputGeometry(
            shape.ImgSize[i], shape.Channels[i], p.FilterSize[i], p.Stride[i], p.PaddingStart[i],
            p.Groups[i], p.RandSparse[i])).ToArray();
        var result = new ScalarConvolution(layer.Name, inputs, p.OutputFilters, p.PartialSum, p.SharedBiases);
        if (result.ModulesX != shape.ModulesX || result.Filters * result.Modules != shape.Outputs)
            throw new ArgumentException($"Layer '{layer.Name}': definition and shape disagree.");
        return result;
    }

    /// <summary>Whole layer in declared input/weight order. First write overwrites and adds bias;
    /// later writes accumulate without adding bias again (src/layer.cu:157-165,833-839).
    /// All buffers are validated before the first write.</summary>
    public void Forward(IReadOnlyList<FloatMatrix> images, IReadOnlyList<FloatMatrix> weights,
        FloatMatrix biases, FloatMatrix target)
    {
        if (images.Count != Inputs.Count || weights.Count != Inputs.Count)
            throw new ArgumentException($"Layer '{Name}': one image and weight matrix per input required.");
        for (int i = 0; i < Inputs.Count; i++) ValidateForward(i, images[i], weights[i], biases, target);
        for (int i = 0; i < Inputs.Count; i++) ForwardInput(i, images[i], weights[i], biases, target, i == 0 ? 0 : 1);
    }

    /// <summary>One input contribution. Bias is added exactly when scaleTargets is zero,
    /// as in the original; a zero scale discards old targets including NaNs.</summary>
    public void ForwardInput(int input, FloatMatrix images, FloatMatrix weights, FloatMatrix biases,
        FloatMatrix target, float scaleTargets = 0)
    {
        ValidateForward(input, images, weights, biases, target);
        var g = Inputs[input];
        var x = images.BackingArray;
        var w = weights.BackingArray;
        for (int f = 0; f < Filters; f++)
            for (int my = 0; my < ModulesX; my++)
                for (int mx = 0; mx < ModulesX; mx++)
                    for (int n = 0; n < images.Cols; n++)
                    {
                        float sum = 0;
                        for (int c = 0; c < g.Channels; c++)
                            for (int fy = 0; fy < g.FilterSize; fy++)
                                for (int fx = 0; fx < g.FilterSize; fx++)
                                {
                                    int y = g.PaddingStart + my * g.Stride + fy;
                                    int xx = g.PaddingStart + mx * g.Stride + fx;
                                    float value = y >= 0 && y < g.ImageSize && xx >= 0 && xx < g.ImageSize
                                        ? x[images.StorageOffset + ((c * g.ImageSize + y) * g.ImageSize + xx) * images.RowStride + n] : 0;
                                    int k = (c * g.FilterSize + fy) * g.FilterSize + fx;
                                    sum += value * w[weights.StorageOffset + k * weights.RowStride + f];
                                }
                        int row = (f * ModulesX + my) * ModulesX + mx;
                        Store(target, row, n, sum, scaleTargets);
                        if (scaleTargets == 0) target[row, n] += biases[SharedBiases ? f : row, 0];
                    }
    }

    /// <summary>Adjoint for one input. The graph calls this only for inputs needing a gradient;
    /// conv1's data gradient is omitted by its caller. No batch averaging of activity gradients.</summary>
    public void BackwardInput(int input, FloatMatrix outputGradient, FloatMatrix weights,
        FloatMatrix target, float scaleTargets = 0)
    {
        var g = Geometry(input);
        ValidateOutput(outputGradient);
        ValidateImage(target, g, outputGradient.Cols);
        ValidateWeights(weights, g);
        Disjoint(target, outputGradient, weights);
        var dy = outputGradient.BackingArray;
        var w = weights.BackingArray;
        for (int c = 0; c < g.Channels; c++)
            for (int y = 0; y < g.ImageSize; y++)
                for (int x = 0; x < g.ImageSize; x++)
                    for (int n = 0; n < target.Cols; n++)
                    {
                        float sum = 0;
                        int minY = Math.Max(0, (int)Math.Ceiling((y - g.PaddingStart - g.FilterSize + 1.0) / g.Stride));
                        int maxY = Math.Min(ModulesX - 1, (y - g.PaddingStart) / g.Stride);
                        int minX = Math.Max(0, (int)Math.Ceiling((x - g.PaddingStart - g.FilterSize + 1.0) / g.Stride));
                        int maxX = Math.Min(ModulesX - 1, (x - g.PaddingStart) / g.Stride);
                        for (int my = minY; my <= maxY; my++)
                            for (int mx = minX; mx <= maxX; mx++)
                            {
                                int fy = y - g.PaddingStart - my * g.Stride;
                                int fx = x - g.PaddingStart - mx * g.Stride;
                                int k = (c * g.FilterSize + fy) * g.FilterSize + fx;
                                int m = my * ModulesX + mx;
                                for (int f = 0; f < Filters; f++)
                                    sum += dy[outputGradient.StorageOffset + (f * Modules + m) * outputGradient.RowStride + n] *
                                        w[weights.StorageOffset + k * weights.RowStride + f];
                            }
                        Store(target, (c * g.ImageSize + y) * g.ImageSize + x, n, sum, scaleTargets);
                    }
    }

    /// <summary>Training divides by normalizationCases (default actual cases), PASS_GC does not.
    /// Supply virtual batch count from GetNumCases for deferred updates. Chunk temporaries always
    /// overwrite; scaleTargets applies once to the final weight gradient (src/layer.cu:860-877).
    /// Chunking deliberately changes float32 association and scales BEFORE final reduction.</summary>
    public void BackwardWeights(int input, FloatMatrix images, FloatMatrix outputGradient,
        FloatMatrix target, float scaleTargets = 0, bool gradientCheck = false, int? normalizationCases = null)
    {
        var g = Geometry(input);
        ValidateOutput(outputGradient);
        ValidateImage(images, g, outputGradient.Cols);
        ValidateWeights(target, g);
        Disjoint(target, images, outputGradient);
        float scale = GradientScale(outputGradient.Cols, gradientCheck, normalizationCases);
        int chunkSize = PartialSum == 0 ? Modules : PartialSum;
        var x = images.BackingArray;
        var dy = outputGradient.BackingArray;
        for (int c = 0; c < g.Channels; c++)
            for (int fy = 0; fy < g.FilterSize; fy++)
                for (int fx = 0; fx < g.FilterSize; fx++)
                    for (int f = 0; f < Filters; f++)
                    {
                        float total = 0;
                        for (int start = 0; start < Modules; start += chunkSize)
                        {
                            float chunk = 0;
                            for (int m = start; m < start + chunkSize; m++)
                            {
                                int y = g.PaddingStart + m / ModulesX * g.Stride + fy;
                                int xx = g.PaddingStart + m % ModulesX * g.Stride + fx;
                                for (int n = 0; n < images.Cols; n++)
                                {
                                    float value = y >= 0 && y < g.ImageSize && xx >= 0 && xx < g.ImageSize
                                        ? x[images.StorageOffset + ((c * g.ImageSize + y) * g.ImageSize + xx) * images.RowStride + n] : 0;
                                    chunk += value * dy[outputGradient.StorageOffset + (f * Modules + m) * outputGradient.RowStride + n];
                                }
                            }
                            total += scale * chunk;
                        }
                        Store(target, (c * g.FilterSize + fy) * g.FilterSize + fx, f, total, scaleTargets);
                    }
    }

    public void BackwardBiases(FloatMatrix outputGradient, FloatMatrix target,
        float scaleTargets = 0, bool gradientCheck = false, int? normalizationCases = null)
    {
        ValidateOutput(outputGradient);
        ValidateBias(target);
        Disjoint(target, outputGradient);
        float scale = GradientScale(outputGradient.Cols, gradientCheck, normalizationCases);
        for (int row = 0; row < target.Rows; row++)
        {
            float sum = 0;
            int count = SharedBiases ? Modules : 1;
            for (int m = 0; m < count; m++)
                for (int n = 0; n < outputGradient.Cols; n++)
                    sum += outputGradient[SharedBiases ? row * Modules + m : row, n];
            Store(target, row, 0, scale * sum, scaleTargets);
        }
    }

    private ConvInputGeometry Geometry(int input)
    {
        if ((uint)input >= (uint)Inputs.Count) throw new ArgumentOutOfRangeException(nameof(input));
        return Inputs[input];
    }
    private void ValidateForward(int input, FloatMatrix images, FloatMatrix weights, FloatMatrix biases, FloatMatrix target)
    {
        var g = Geometry(input);
        ValidateOutput(target);
        ValidateImage(images, g, target.Cols);
        ValidateWeights(weights, g);
        ValidateBias(biases);
        Disjoint(target, images, weights, biases);
    }
    private void ValidateOutput(FloatMatrix m) => Validate(m, checked(Filters * Modules), m.Cols);
    private void ValidateImage(FloatMatrix m, ConvInputGeometry g, int cases) => Validate(m, checked(g.Channels * g.ImageSize * g.ImageSize), cases);
    private void ValidateWeights(FloatMatrix m, ConvInputGeometry g) => Validate(m, checked(g.Channels * g.FilterSize * g.FilterSize), Filters);
    private void ValidateBias(FloatMatrix m) => Validate(m, SharedBiases ? Filters : checked(Filters * Modules), 1);
    private void Validate(FloatMatrix m, int rows, int cols)
    {
        if (m.Layout != MatrixLayout.RowMajor || m.ColumnStride != 1 || m.Rows != rows || m.Cols != cols || cols <= 0)
            throw new ArgumentException($"Layer '{Name}': expected canonical C-order ({rows},{cols}) matrix.");
    }
    private void Disjoint(FloatMatrix target, params FloatMatrix[] sources)
    {
        if (sources.Any(s => ReferenceEquals(s.BackingArray, target.BackingArray)))
            throw new ArgumentException($"Layer '{Name}': convolution target must not share operand storage.");
    }
    private static float GradientScale(int cases, bool gradientCheck, int? normalizationCases)
    {
        int count = normalizationCases ?? cases;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        return gradientCheck ? 1 : 1.0f / count;
    }
    private static void Store(FloatMatrix target, int row, int col, float value, float scaleTargets) =>
        target[row, col] = scaleTargets == 0 ? value : scaleTargets * target[row, col] + value;
}
