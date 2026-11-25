using System.Numerics;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Conv;

/// <summary>Managed SIMD convolution. All reductions retain the scalar oracle order;
/// lanes and parallel tasks own independent outputs. No fused multiply-add or horizontal sums.
/// Forward uses 64-position im2col panels; weight gradients use at most 4096 reduction
/// positions per transposed upstream panel. See ANX-6_REPORT.md for workspace bounds.
/// Original src/layer.cu:823-895; A:src/cudaconv2/filter_acts.cu:129-154.</summary>
public sealed class OptimizedConvolution
{
    public const int ForwardTile = 64;
    public const int ReductionTile = 4096;
    public int MaxDegreeOfParallelism { get; }
    private ParallelOptions Options => new() { MaxDegreeOfParallelism = MaxDegreeOfParallelism };
    public string Name { get; }
    public IReadOnlyList<ConvInputGeometry> Inputs { get; }
    public int Filters { get; }
    public int ModulesX { get; }
    public int Modules => checked(ModulesX * ModulesX);
    public int PartialSum { get; }
    public bool SharedBiases { get; }

    public OptimizedConvolution(string name, IReadOnlyList<ConvInputGeometry> inputs, int filters,
        int partialSum = 0, bool sharedBiases = true, int maxDegreeOfParallelism = -1)
    {
        if (maxDegreeOfParallelism != -1 && maxDegreeOfParallelism <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism));
        MaxDegreeOfParallelism = maxDegreeOfParallelism;
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

    public static OptimizedConvolution FromLayer(LayerDefinition layer, LayerShape shape, int maxDegreeOfParallelism = -1)
    {
        if (layer.TypedParams is not ConvLayerParams p || shape.Type != "conv" || layer.Name != shape.Name)
            throw new ArgumentException("Matching convolution definition and shape required.");
        var inputs = Enumerable.Range(0, layer.Inputs.Count).Select(i => new ConvInputGeometry(
            shape.ImgSize[i], shape.Channels[i], p.FilterSize[i], p.Stride[i], p.PaddingStart[i],
            p.Groups[i], p.RandSparse[i])).ToArray();
        var result = new OptimizedConvolution(layer.Name, inputs, p.OutputFilters, p.PartialSum, p.SharedBiases, maxDegreeOfParallelism);
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
        int cases = images.Cols, kCount = g.Channels * g.FilterSize * g.FilterSize;
        int positions = checked(Modules * cases), tiles = (positions - 1) / ForwardTile + 1;
        var x = images.BackingArray; var w = weights.BackingArray;
        // Explicit worker partition bounds live panels by the requested worker count.
        int workers = Math.Min(tiles, WorkerCount);
        Parallel.For(0, workers, Options, worker =>
        {
            var panel = new float[checked(kCount * ForwardTile)];
            for (int tile = worker; tile < tiles; tile += workers)
            {
                int start = tile * ForwardTile, length = Math.Min(ForwardTile, positions - start);
                for (int k = 0; k < kCount; k++)
                {
                    int c = k / (g.FilterSize * g.FilterSize);
                    int fy = k / g.FilterSize % g.FilterSize, fx = k % g.FilterSize;
                    for (int t = 0; t < length; t++)
                    {
                        int p = start + t, m = p / cases, n = p % cases;
                        int y = g.PaddingStart + m / ModulesX * g.Stride + fy;
                        int xx = g.PaddingStart + m % ModulesX * g.Stride + fx;
                        panel[k * ForwardTile + t] = y >= 0 && y < g.ImageSize && xx >= 0 && xx < g.ImageSize
                            ? x[images.StorageOffset + ((c * g.ImageSize + y) * g.ImageSize + xx) * images.RowStride + n] : 0;
                    }
                }
                for (int f = 0; f < Filters; f++)
                {
                    int t = 0;
                    for (; t + Vector<float>.Count <= length; t += Vector<float>.Count)
                    {
                        var sum = Vector<float>.Zero;
                        for (int k = 0; k < kCount; k++)
                            sum += new Vector<float>(panel, k * ForwardTile + t) *
                                new Vector<float>(w[weights.StorageOffset + k * weights.RowStride + f]);
                        // A tile can cross strided matrix rows; scatter only the completed lanes.
                        for (int lane = 0; lane < Vector<float>.Count; lane++)
                            StoreForward(f, start + t + lane, sum[lane]);
                    }
                    for (; t < length; t++)
                    {
                        float sum = 0;
                        for (int k = 0; k < kCount; k++)
                            sum += panel[k * ForwardTile + t] * w[weights.StorageOffset + k * weights.RowStride + f];
                        StoreForward(f, start + t, sum);
                    }
                }
            }
        });
        void StoreForward(int f, int p, float value)
        {
            int row = f * Modules + p / cases, n = p % cases;
            Store(target, row, n, value, scaleTargets);
            if (scaleTargets == 0) target[row, n] += biases[SharedBiases ? f : row, 0];
        }
    }

    private int WorkerCount => MaxDegreeOfParallelism == -1 ? Environment.ProcessorCount : MaxDegreeOfParallelism;

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
        var dy = outputGradient.BackingArray; var w = weights.BackingArray;
        int rows = target.Rows, cases = target.Cols;
        Parallel.For(0, rows, Options, row =>
        {
            int c = row / (g.ImageSize * g.ImageSize), y = row / g.ImageSize % g.ImageSize, x = row % g.ImageSize;
            int minY = Math.Max(0, (int)Math.Ceiling((y - g.PaddingStart - g.FilterSize + 1.0) / g.Stride));
            int maxY = Math.Min(ModulesX - 1, (y - g.PaddingStart) / g.Stride);
            int minX = Math.Max(0, (int)Math.Ceiling((x - g.PaddingStart - g.FilterSize + 1.0) / g.Stride));
            int maxX = Math.Min(ModulesX - 1, (x - g.PaddingStart) / g.Stride);
            int n = 0;
            for (; n + Vector<float>.Count <= cases; n += Vector<float>.Count)
            {
                var sum = Vector<float>.Zero;
                for (int my = minY; my <= maxY; my++)
                    for (int mx = minX; mx <= maxX; mx++)
                    {
                        int fy = y - g.PaddingStart - my * g.Stride, fx = x - g.PaddingStart - mx * g.Stride;
                        int k = (c * g.FilterSize + fy) * g.FilterSize + fx, m = my * ModulesX + mx;
                        for (int f = 0; f < Filters; f++)
                            sum += new Vector<float>(dy, outputGradient.StorageOffset + (f * Modules + m) * outputGradient.RowStride + n) *
                                new Vector<float>(w[weights.StorageOffset + k * weights.RowStride + f]);
                    }
                int offset = target.StorageOffset + row * target.RowStride + n;
                if (scaleTargets != 0) sum += new Vector<float>(target.BackingArray, offset) * new Vector<float>(scaleTargets);
                sum.CopyTo(target.BackingArray, offset);
            }
            for (; n < cases; n++)
            {
                float sum = 0;
                for (int my = minY; my <= maxY; my++)
                    for (int mx = minX; mx <= maxX; mx++)
                    {
                        int fy = y - g.PaddingStart - my * g.Stride, fx = x - g.PaddingStart - mx * g.Stride;
                        int k = (c * g.FilterSize + fy) * g.FilterSize + fx, m = my * ModulesX + mx;
                        for (int f = 0; f < Filters; f++)
                            sum += dy[outputGradient.StorageOffset + (f * Modules + m) * outputGradient.RowStride + n] *
                                w[weights.StorageOffset + k * weights.RowStride + f];
                    }
                Store(target, row, n, sum, scaleTargets);
            }
        });
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
        int cases = images.Cols, kCount = target.Rows;
        int positions = checked(Modules * cases), chunkPositions = checked(chunkSize * cases);
        int tileSize = Math.Min(ReductionTile, positions);
        var panel = new float[checked(tileSize * Filters)];
        var chunks = new float[checked(kCount * Filters)];
        var totals = new float[chunks.Length];
        int workers = Math.Min(kCount, WorkerCount);
        var imagePanels = Enumerable.Range(0, workers).Select(_ => new float[tileSize]).ToArray();
        var x = images.BackingArray; var dy = outputGradient.BackingArray;
        for (int start = 0; start < positions; start += tileSize)
        {
            int length = Math.Min(tileSize, positions - start);
            // Transpose upstream once per bounded reduction panel: (filter,module,case) -> (module,case,filter).
            for (int f = 0; f < Filters; f++)
                for (int t = 0; t < length; t++)
                {
                    int p = start + t;
                    panel[t * Filters + f] = dy[outputGradient.StorageOffset + (f * Modules + p / cases) * outputGradient.RowStride + p % cases];
                }
            Parallel.For(0, workers, Options, worker =>
            {
                var imagePanel = imagePanels[worker];
                for (int k = worker; k < kCount; k += workers)
                {
                    int c = k / (g.FilterSize * g.FilterSize), fy = k / g.FilterSize % g.FilterSize, fx = k % g.FilterSize;
                    // Compute window coordinates once per k/position, reused by all filter vectors.
                    for (int t = 0; t < length; t++)
                    {
                        int p = start + t, m = p / cases;
                        int y = g.PaddingStart + m / ModulesX * g.Stride + fy;
                        int xx = g.PaddingStart + m % ModulesX * g.Stride + fx;
                        imagePanel[t] = y >= 0 && y < g.ImageSize && xx >= 0 && xx < g.ImageSize
                            ? x[images.StorageOffset + ((c * g.ImageSize + y) * g.ImageSize + xx) * images.RowStride + p % cases] : 0;
                    }
                    int f = 0;
                    for (; f + Vector<float>.Count <= Filters; f += Vector<float>.Count)
                    {
                        int offset = k * Filters + f;
                        var chunk = new Vector<float>(chunks, offset);
                        var total = new Vector<float>(totals, offset);
                        for (int t = 0; t < length;)
                        {
                            // Panel boundaries never become reduction boundaries. Only partialSum does.
                            int remaining = chunkPositions - (start + t) % chunkPositions;
                            int end = t + Math.Min(remaining, length - t);
                            bool complete = remaining <= length - t;
                            for (; t < end; t++)
                                chunk += new Vector<float>(imagePanel[t]) * new Vector<float>(panel, t * Filters + f);
                            if (complete) { total += new Vector<float>(scale) * chunk; chunk = Vector<float>.Zero; }
                        }
                        chunk.CopyTo(chunks, offset); total.CopyTo(totals, offset);
                    }
                    for (; f < Filters; f++)
                    {
                        int offset = k * Filters + f;
                        float chunk = chunks[offset], total = totals[offset];
                        for (int t = 0; t < length;)
                        {
                            int remaining = chunkPositions - (start + t) % chunkPositions;
                            int end = t + Math.Min(remaining, length - t);
                            bool complete = remaining <= length - t;
                            for (; t < end; t++) chunk += imagePanel[t] * panel[t * Filters + f];
                            if (complete) { total += scale * chunk; chunk = 0; }
                        }
                        chunks[offset] = chunk; totals[offset] = total;
                    }
                }
            });
        }
        for (int k = 0; k < kCount; k++)
            for (int f = 0; f < Filters; f++) Store(target, k, f, totals[k * Filters + f], scaleTargets);
    }

    public void BackwardBiases(FloatMatrix outputGradient, FloatMatrix target,
        float scaleTargets = 0, bool gradientCheck = false, int? normalizationCases = null)
    {
        ValidateOutput(outputGradient);
        ValidateBias(target);
        Disjoint(target, outputGradient);
        float scale = GradientScale(outputGradient.Cols, gradientCheck, normalizationCases);
        var dy = outputGradient.BackingArray;
        Parallel.For(0, target.Rows, Options, row =>
        {
            float sum = 0;
            int count = SharedBiases ? Modules : 1;
            for (int m = 0; m < count; m++)
            {
                int offset = outputGradient.StorageOffset + (SharedBiases ? row * Modules + m : row) * outputGradient.RowStride;
                for (int n = 0; n < outputGradient.Cols; n++) sum += dy[offset + n];
            }
            Store(target, row, 0, scale * sum, scaleTargets);
        });
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
