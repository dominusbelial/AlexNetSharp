using System.Numerics;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Pool;

/// <summary>Independent SIMD implementation of ScalarPool's contract. Parallel rows
/// own their writes; SIMD lanes are independent cases. No horizontal reductions,
/// atomics or shared accumulators; fixed window order at every worker count.</summary>
public sealed class OptimizedPool
{
    public string Name { get; }
    public PoolGeometry Geometry { get; }
    public string Pool { get; }
    private readonly ParallelOptions _options;
    public OptimizedPool(string name, PoolGeometry geometry, string pool = "max", int maxDegreeOfParallelism = -1)
    {
        PoolContract.Geometry(name, geometry, pool);
        if (maxDegreeOfParallelism == 0 || maxDegreeOfParallelism < -1)
            throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism));
        Name = name; Geometry = geometry; Pool = pool;
        _options = new() { MaxDegreeOfParallelism = maxDegreeOfParallelism };
    }
    public static OptimizedPool FromLayer(LayerDefinition layer, LayerShape shape, int maxDegreeOfParallelism = -1) =>
        new(layer.Name, PoolContract.FromLayer(layer, shape), ((PoolLayerParams)layer.TypedParams).Pool, maxDegreeOfParallelism);

    public void Forward(FloatMatrix input, FloatMatrix target)
    {
        var g = Geometry;
        PoolContract.Matrix(Name, input, g.InputRows, target.Cols);
        PoolContract.Matrix(Name, target, g.OutputRows, input.Cols);
        PoolContract.Disjoint(Name, target, input);
        var src = input.BackingArray; var dst = target.BackingArray;
        Parallel.For(0, target.Rows, _options, row =>
        {
            int c = row / (g.OutputSize * g.OutputSize), oy = row / g.OutputSize % g.OutputSize, ox = row % g.OutputSize;
            int sy = g.Start + oy * g.Stride, sx = g.Start + ox * g.Stride;
            int ly = Math.Max(0, sy), lx = Math.Max(0, sx);
            int ey = Math.Min(g.ImageSize, sy + g.SizeX), ex = Math.Min(g.ImageSize, sx + g.SizeX);
            int area = (ey - ly) * (ex - lx), n = 0, dest = target.StorageOffset + row * target.RowStride;
            for (; n <= input.Cols - Vector<float>.Count; n += Vector<float>.Count)
            {
                var acc = new Vector<float>(Pool == "max" ? -2e38f : 0);
                for (int y = ly; y < ey; y++)
                    for (int x = lx; x < ex; x++)
                    {
                        var b = new Vector<float>(src, input.StorageOffset + ((c * g.ImageSize + y) * g.ImageSize + x) * input.RowStride + n);
                        if (Pool == "avg") acc += b;
                        else
                        {
                            var zeros = Vector.BitwiseAnd(Vector.Equals(acc, Vector<float>.Zero), Vector.Equals(b, Vector<float>.Zero));
                            var maximum = Vector.ConditionalSelect(Vector.GreaterThan(b, acc), b, acc);
                            acc = Vector.ConditionalSelect(zeros, Vector.BitwiseAnd(acc, b), maximum);
                        }
                    }
                if (Pool == "avg") acc /= new Vector<float>(area);
                acc.CopyTo(dst, dest + n);
            }
            for (; n < input.Cols; n++)
            {
                float acc = Pool == "max" ? -2e38f : 0;
                for (int y = ly; y < ey; y++)
                    for (int x = lx; x < ex; x++)
                    {
                        float b = src[input.StorageOffset + ((c * g.ImageSize + y) * g.ImageSize + x) * input.RowStride + n];
                        acc = Pool == "avg" ? acc + b : float.MaxNumber(acc, b);
                    }
                dst[dest + n] = Pool == "avg" ? acc / area : acc;
            }
        });
    }

    public void Backward(FloatMatrix? input, FloatMatrix outputGradient, FloatMatrix? output,
        FloatMatrix target, float scaleTargets = 0)
    {
        var g = Geometry;
        PoolContract.Matrix(Name, outputGradient, g.OutputRows, target.Cols);
        PoolContract.Matrix(Name, target, g.InputRows, outputGradient.Cols);
        PoolContract.Disjoint(Name, target, outputGradient);
        if (Pool == "max")
        {
            ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(output);
            PoolContract.Matrix(Name, input, g.InputRows, target.Cols);
            PoolContract.Matrix(Name, output, g.OutputRows, target.Cols);
            PoolContract.Disjoint(Name, target, input, output);
        }
        var dy = outputGradient.BackingArray; var dst = target.BackingArray;
        Parallel.For(0, target.Rows, _options, row =>
        {
            int c = row / (g.ImageSize * g.ImageSize), y = row / g.ImageSize % g.ImageSize, x = row % g.ImageSize;
            int minY = Math.Max(0, (int)Math.Ceiling((y - (double)g.Start - g.SizeX + 1) / g.Stride));
            int minX = Math.Max(0, (int)Math.Ceiling((x - (double)g.Start - g.SizeX + 1) / g.Stride));
            int endY = Math.Min(g.OutputSize, (int)Math.Floor((y - (double)g.Start) / g.Stride) + 1);
            int endX = Math.Min(g.OutputSize, (int)Math.Floor((x - (double)g.Start) / g.Stride) + 1);
            int dest = target.StorageOffset + row * target.RowStride, n = 0;
            for (; n <= target.Cols - Vector<float>.Count; n += Vector<float>.Count)
            {
                var sum = Vector<float>.Zero;
                for (int oy = minY; oy < endY; oy++)
                    for (int ox = minX; ox < endX; ox++)
                    {
                        int o = (c * g.OutputSize + oy) * g.OutputSize + ox;
                        var grad = new Vector<float>(dy, outputGradient.StorageOffset + o * outputGradient.RowStride + n);
                        Vector<float> factor;
                        if (Pool == "max")
                        {
                            var a = new Vector<float>(input!.BackingArray, input.StorageOffset + row * input.RowStride + n);
                            var b = new Vector<float>(output!.BackingArray, output.StorageOffset + o * output.RowStride + n);
                            factor = Vector.ConditionalSelect(Vector.Equals(a, b), Vector<float>.One, Vector<float>.Zero);
                        }
                        else factor = new Vector<float>(1f / Area(oy, ox));
                        sum += factor * grad;
                    }
                if (scaleTargets != 0) sum = new Vector<float>(scaleTargets) * new Vector<float>(dst, dest + n) + sum;
                sum.CopyTo(dst, dest + n);
            }
            for (; n < target.Cols; n++)
            {
                float sum = 0;
                for (int oy = minY; oy < endY; oy++)
                    for (int ox = minX; ox < endX; ox++)
                    {
                        int o = (c * g.OutputSize + oy) * g.OutputSize + ox;
                        float factor = Pool == "max" ? (input![row, n] == output![o, n] ? 1f : 0f) : 1f / Area(oy, ox);
                        sum += factor * dy[outputGradient.StorageOffset + o * outputGradient.RowStride + n];
                    }
                dst[dest + n] = scaleTargets == 0 ? sum : scaleTargets * dst[dest + n] + sum;
            }
        });
        int Area(int oy, int ox)
        {
            int sy = g.Start + oy * g.Stride, sx = g.Start + ox * g.Stride;
            return (Math.Min(g.ImageSize, sy + g.SizeX) - Math.Max(0, sy)) *
                (Math.Min(g.ImageSize, sx + g.SizeX) - Math.Max(0, sx));
        }
    }
}
