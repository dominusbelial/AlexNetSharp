using System.Numerics;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Norm;

/// <summary>Independent SIMD/parallel norm. Each row owns its writes; SIMD lanes
/// are cases. Ordered window tables contain channel numbers or spatial indices.
/// No horizontal reductions, sliding sums, FMA, atomics or cross-worker reductions.
/// MathF.Pow is evaluated per case, identically to the scalar reference.</summary>
public sealed class OptimizedNorm
{
    public string Name { get; }
    public NormGeometry Geometry { get; }
    public string Kind { get; }
    public float AddScale { get; }
    public float Power { get; }
    public float MinDiv { get; }
    public bool Blocked { get; }
    private readonly ParallelOptions _options;
    private readonly int[][] _forward, _backward;

    public OptimizedNorm(string name, NormGeometry geometry, string kind, float addScale, float power,
        float minDiv = 1, bool blocked = false, int maxDegreeOfParallelism = -1)
    {
        NormContract.Validate(name, geometry, kind);
        if (maxDegreeOfParallelism == 0 || maxDegreeOfParallelism < -1)
            throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism));
        Name = name; Geometry = geometry; Kind = kind; AddScale = addScale;
        Power = power; MinDiv = minDiv; Blocked = blocked;
        _options = new() { MaxDegreeOfParallelism = maxDegreeOfParallelism };
        int count = kind == "cmrnorm" ? geometry.Channels : geometry.ImageSize * geometry.ImageSize;
        _forward = new int[count][]; _backward = new int[count][];
        for (int i = 0; i < count; i++)
        {
            _forward[i] = Window(i, false);
            _backward[i] = Window(i, true);
        }
    }

    public static OptimizedNorm FromLayer(LayerDefinition layer, LayerShape shape, int maxDegreeOfParallelism = -1)
    {
        var p = NormContract.FromLayer(layer, shape);
        return new(layer.Name, p.Geometry, layer.Type, p.Scale, p.Power, p.MinDiv, p.Blocked, maxDegreeOfParallelism);
    }

    private int[] Window(int index, bool reverse)
    {
        int width = Geometry.ImageSize, half = Geometry.Size / 2;
        int offset = reverse ? half - Geometry.Size + 1 : -half;
        var list = new List<int>();
        if (Kind == "cmrnorm")
        {
            int start = Blocked ? index / Geometry.Size * Geometry.Size : index + offset;
            for (int f = Math.Max(0, start); f < Math.Min(Geometry.Channels, start + Geometry.Size); f++) list.Add(f);
        }
        else
        {
            int sy = index / width + offset, sx = index % width + offset;
            for (int y = Math.Max(0, sy); y < Math.Min(width, sy + Geometry.Size); y++)
                for (int x = Math.Max(0, sx); x < Math.Min(width, sx + Geometry.Size); x++) list.Add(y * width + x);
        }
        return list.ToArray();
    }

    public void Forward(FloatMatrix input, FloatMatrix denoms, FloatMatrix target)
    {
        int rows = Geometry.Rows, plane = Geometry.ImageSize * Geometry.ImageSize;
        NormContract.Matrix(Name, input, rows, target.Cols);
        NormContract.Matrix(Name, denoms, rows, input.Cols);
        NormContract.Matrix(Name, target, rows, input.Cols);
        NormContract.Disjoint(Name, target, input, denoms); NormContract.Disjoint(Name, denoms, input);
        var src = input.BackingArray; var cache = denoms.BackingArray; var dst = target.BackingArray;
        Parallel.For(0, rows, _options, row =>
        {
            int channel = row / plane, pixel = row % plane;
            var window = _forward[Kind == "cmrnorm" ? channel : pixel];
            int d = denoms.StorageOffset + row * denoms.RowStride, o = target.StorageOffset + row * target.RowStride;
            int origin = input.StorageOffset + row * input.RowStride, n = 0;
            for (; n <= input.Cols - Vector<float>.Count; n += Vector<float>.Count)
            {
                var sum = Vector<float>.Zero;
                foreach (int i in window)
                {
                    int r = Kind == "cmrnorm" ? i * plane + pixel : channel * plane + i;
                    var v = new Vector<float>(src, input.StorageOffset + r * input.RowStride + n);
                    sum += v * v;
                }
                (new Vector<float>(Kind == "cmrnorm" ? MinDiv : 1f) + new Vector<float>(AddScale) * sum).CopyTo(cache, d + n);
            }
            for (; n < input.Cols; n++)
            {
                float sum = 0;
                foreach (int i in window)
                {
                    int r = Kind == "cmrnorm" ? i * plane + pixel : channel * plane + i;
                    float v = src[input.StorageOffset + r * input.RowStride + n]; sum += v * v;
                }
                cache[d + n] = (Kind == "cmrnorm" ? MinDiv : 1f) + AddScale * sum;
            }
            for (n = 0; n < input.Cols; n++) dst[o + n] = src[origin + n] * MathF.Pow(cache[d + n], -Power);
        });
    }

    public void Backward(FloatMatrix input, FloatMatrix outputGradient, FloatMatrix denoms,
        FloatMatrix output, FloatMatrix target, float scaleTargets = 0)
    {
        int rows = Geometry.Rows, plane = Geometry.ImageSize * Geometry.ImageSize, cases = input.Cols;
        foreach (var m in new[] { input, outputGradient, denoms, output, target }) NormContract.Matrix(Name, m, rows, cases);
        NormContract.Disjoint(Name, target, input, outputGradient, denoms, output);
        var tmp = new float[checked(rows * cases)];
        var grad = outputGradient.BackingArray; var acts = output.BackingArray; var cache = denoms.BackingArray;
        var src = input.BackingArray; var dst = target.BackingArray;
        float coefficient = -2f * AddScale * Power;
        Parallel.For(0, rows, _options, row =>
        {
            int a = output.StorageOffset + row * output.RowStride, d = denoms.StorageOffset + row * denoms.RowStride;
            int g = outputGradient.StorageOffset + row * outputGradient.RowStride, t = row * cases, n = 0;
            for (; n <= cases - Vector<float>.Count; n += Vector<float>.Count)
                (new Vector<float>(coefficient) * new Vector<float>(grad, g + n) * new Vector<float>(acts, a + n) /
                    new Vector<float>(cache, d + n)).CopyTo(tmp, t + n);
            for (; n < cases; n++) tmp[t + n] = coefficient * grad[g + n] * acts[a + n] / cache[d + n];
        });
        Parallel.For(0, rows, _options, row =>
        {
            int channel = row / plane, pixel = row % plane;
            var window = _backward[Kind == "cmrnorm" ? channel : pixel];
            int d = denoms.StorageOffset + row * denoms.RowStride, o = target.StorageOffset + row * target.RowStride;
            int g = outputGradient.StorageOffset + row * outputGradient.RowStride, origin = input.StorageOffset + row * input.RowStride;
            // Ordered SIMD sums feed the scalar pow/final arithmetic per lane.
            int n = 0;
            for (; n <= cases - Vector<float>.Count; n += Vector<float>.Count)
            {
                var sum = Vector<float>.Zero;
                foreach (int i in window)
                {
                    int r = Kind == "cmrnorm" ? i * plane + pixel : channel * plane + i;
                    sum += new Vector<float>(tmp, r * cases + n);
                }
                for (int lane = 0; lane < Vector<float>.Count; lane++) Write(n + lane, sum[lane]);
            }
            for (; n < cases; n++)
            {
                float sum = 0;
                foreach (int i in window)
                {
                    int r = Kind == "cmrnorm" ? i * plane + pixel : channel * plane + i;
                    sum += tmp[r * cases + n];
                }
                Write(n, sum);
            }
            void Write(int col, float sum)
            {
                float value = src[origin + col] * sum + grad[g + col] * MathF.Pow(cache[d + col], -Power);
                dst[o + col] = scaleTargets == 0 ? value : scaleTargets * dst[o + col] + value;
            }
        });
    }
}
