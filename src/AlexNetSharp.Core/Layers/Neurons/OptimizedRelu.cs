using System.Numerics;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Neurons;

/// <summary>Independent SIMD ReLU; contiguous dimension is vectorized for C/F
/// layouts, strided operands use independent scalar elements. In-place safe.
/// Parallel lines own writes; masks multiply upstream, preserving NaN behavior.</summary>
public static class OptimizedRelu
{
    public static void Forward(FloatMatrix input, FloatMatrix target, int maxDegreeOfParallelism = -1)
    {
        ReluContract.Validate(target, input);
        Execute(input, null, target, 0, maxDegreeOfParallelism);
    }
    public static void Backward(FloatMatrix output, FloatMatrix outputGradient, FloatMatrix target,
        float scaleTargets = 0, int maxDegreeOfParallelism = -1)
    {
        ReluContract.Validate(target, output, outputGradient);
        Execute(output, outputGradient, target, scaleTargets, maxDegreeOfParallelism);
    }
    private static void Execute(FloatMatrix source, FloatMatrix? gradient, FloatMatrix target, float scale, int threads)
    {
        if (threads == 0 || threads < -1) throw new ArgumentOutOfRangeException(nameof(threads));
        bool c = target.Layout == MatrixLayout.RowMajor;
        int lines = c ? target.Rows : target.Cols, length = c ? target.Cols : target.Rows;
        bool vector = (c ? target.ColumnStride : target.RowStride) == 1 &&
            (c ? source.ColumnStride : source.RowStride) == 1 &&
            (gradient == null || (c ? gradient.ColumnStride : gradient.RowStride) == 1);
        Parallel.For(0, lines, new ParallelOptions { MaxDegreeOfParallelism = threads }, line =>
        {
            int n = 0;
            int si = source.StorageOffset + line * (c ? source.RowStride : source.ColumnStride);
            int ti = target.StorageOffset + line * (c ? target.RowStride : target.ColumnStride);
            int gi = gradient == null ? 0 : gradient.StorageOffset + line * (c ? gradient.RowStride : gradient.ColumnStride);
            if (vector)
                for (; n <= length - Vector<float>.Count; n += Vector<float>.Count)
                {
                    var x = new Vector<float>(source.BackingArray, si + n);
                    var value = gradient == null ? Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero), Vector<float>.Zero, x) :
                        new Vector<float>(gradient.BackingArray, gi + n) * Vector.ConditionalSelect(Vector.GreaterThan(x, Vector<float>.Zero), Vector<float>.One, Vector<float>.Zero);
                    if (gradient != null && scale > 0) value += new Vector<float>(target.BackingArray, ti + n);
                    value.CopyTo(target.BackingArray, ti + n);
                }
            for (; n < length; n++)
            {
                int r = c ? line : n, col = c ? n : line;
                float x = source[r, col];
                float value = gradient == null ? (x < 0 ? 0 : x) : gradient[r, col] * (x > 0 ? 1f : 0f);
                target[r, col] = gradient != null && scale > 0 ? value + target[r, col] : value;
            }
        });
    }
}
