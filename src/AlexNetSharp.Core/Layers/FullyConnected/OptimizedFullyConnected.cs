using System.Numerics;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.FullyConnected;

/// <summary>Managed SIMD with independent output ownership and ordered reductions.
/// No horizontal reductions, atomics or FMA. Forward/dX allocate no float workspace;
/// dW packs one (width,cases) input transpose: 4*width*cases bytes, shared by all workers.
/// Input fan-in is sequential so peak workspace is the largest single input.</summary>
public sealed class OptimizedFullyConnected
{
    public FcGeometry Geometry { get; }
    public int MaxDegreeOfParallelism { get; }
    private ParallelOptions Options => new() { MaxDegreeOfParallelism = MaxDegreeOfParallelism };
    public OptimizedFullyConnected(FcGeometry geometry, int maxDegreeOfParallelism = -1)
    {
        if (maxDegreeOfParallelism != -1 && maxDegreeOfParallelism <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism));
        Geometry = geometry; MaxDegreeOfParallelism = maxDegreeOfParallelism;
    }
    public static OptimizedFullyConnected FromLayer(LayerDefinition layer, LayerShape shape, int maxDegreeOfParallelism = -1) =>
        new(FcGeometry.FromLayer(layer, shape), maxDegreeOfParallelism);

    public void Forward(IReadOnlyList<FloatMatrix> inputs, IReadOnlyList<FloatMatrix> weights, FloatMatrix bias, FloatMatrix target)
    {
        Geometry.ForwardAll(inputs, weights, bias, target);
        for (int i = 0; i < inputs.Count; i++) ForwardInput(i, inputs[i], weights[i], bias, target, i == 0 ? 0 : 1);
    }
    public void ForwardInput(int input, FloatMatrix x, FloatMatrix w, FloatMatrix bias, FloatMatrix target, float scaleTargets = 0)
    {
        Geometry.Forward(input, x, w, bias, target);
        var xa = x.BackingArray; var wa = w.BackingArray; var ya = target.BackingArray;
        int cases = x.Rows, width = x.Cols;
        Parallel.For(0, Geometry.Outputs, Options, o =>
        {
            int wb = w.StorageOffset + o * w.ColumnStride, yb = target.StorageOffset + o * target.ColumnStride;
            float b = bias[0, o]; int n = 0;
            for (; n + Vector<float>.Count <= cases; n += Vector<float>.Count)
            {
                var sum = Vector<float>.Zero;
                for (int k = 0; k < width; k++)
                    sum += new Vector<float>(xa, x.StorageOffset + k * x.ColumnStride + n) * new Vector<float>(wa[wb + k]);
                if (scaleTargets != 0) sum = new Vector<float>(scaleTargets) * new Vector<float>(ya, yb + n) + sum;
                else sum += new Vector<float>(b);
                sum.CopyTo(ya, yb + n);
            }
            for (; n < cases; n++)
            {
                float sum = 0;
                for (int k = 0; k < width; k++) sum += xa[x.StorageOffset + k * x.ColumnStride + n] * wa[wb + k];
                if (scaleTargets != 0) sum = scaleTargets * ya[yb + n] + sum;
                else sum += b;
                ya[yb + n] = sum;
            }
        });
    }
    public void BackwardInput(int input, FloatMatrix dy, FloatMatrix w, FloatMatrix target, float scaleTargets = 0)
    {
        Geometry.Input(input, dy, w, target);
        var da = dy.BackingArray; var wa = w.BackingArray; var ta = target.BackingArray;
        int lanes = Vector<float>.Count, width = target.Cols, tiles = (width - 1) / lanes + 1;
        Parallel.For(0, tiles, Options, tile =>
        {
            int k = tile * lanes;
            for (int n = 0; n < dy.Rows; n++)
            {
                if (k + lanes <= width)
                {
                    var sum = Vector<float>.Zero;
                    for (int o = 0; o < Geometry.Outputs; o++)
                        sum += new Vector<float>(da[dy.StorageOffset + o * dy.ColumnStride + n]) *
                            new Vector<float>(wa, w.StorageOffset + o * w.ColumnStride + k);
                    for (int lane = 0; lane < lanes; lane++)
                    {
                        int at = target.StorageOffset + (k + lane) * target.ColumnStride + n;
                        ta[at] = scaleTargets == 0 ? sum[lane] : scaleTargets * ta[at] + sum[lane];
                    }
                }
                else
                    for (int kk = k; kk < width; kk++)
                    {
                        float sum = 0;
                        for (int o = 0; o < Geometry.Outputs; o++)
                            sum += da[dy.StorageOffset + o * dy.ColumnStride + n] * wa[w.StorageOffset + o * w.ColumnStride + kk];
                        int at = target.StorageOffset + kk * target.ColumnStride + n;
                        ta[at] = scaleTargets == 0 ? sum : scaleTargets * ta[at] + sum;
                    }
            }
        });
    }
    public void BackwardWeights(int input, FloatMatrix x, FloatMatrix dy, FloatMatrix target, float alpha = 1, float beta = 0)
    {
        Geometry.Weight(input, x, dy, target);
        int width = x.Cols, cases = x.Rows;
        // Case-major packed X enables SIMD over independent weight rows. No reduction reassociation.
        var packed = alpha == 0 ? Array.Empty<float>() : new float[checked(width * cases)];
        if (alpha != 0)
            for (int n = 0; n < cases; n++)
                for (int k = 0; k < width; k++) packed[n * width + k] = x[n, k];
        var da = dy.BackingArray; var ta = target.BackingArray;
        Parallel.For(0, Geometry.Outputs, Options, o =>
        {
            int tb = target.StorageOffset + o * target.ColumnStride, db = dy.StorageOffset + o * dy.ColumnStride, k = 0;
            for (; k + Vector<float>.Count <= width; k += Vector<float>.Count)
            {
                var sum = Vector<float>.Zero;
                if (alpha != 0)
                {
                    for (int n = 0; n < cases; n++)
                        sum += new Vector<float>(packed, n * width + k) * new Vector<float>(da[db + n]);
                    sum *= new Vector<float>(alpha);
                }
                if (beta != 0) sum = new Vector<float>(beta) * new Vector<float>(ta, tb + k) + sum;
                sum.CopyTo(ta, tb + k);
            }
            for (; k < width; k++)
            {
                float sum = 0;
                if (alpha != 0)
                {
                    for (int n = 0; n < cases; n++) sum += packed[n * width + k] * da[db + n];
                    sum *= alpha;
                }
                ta[tb + k] = beta == 0 ? sum : beta * ta[tb + k] + sum;
            }
        });
    }
    public void BackwardBiases(FloatMatrix dy, FloatMatrix target, float alpha = 1, float beta = 0)
    {
        Geometry.Bias(dy, target);
        var da = dy.BackingArray;
        Parallel.For(0, Geometry.Outputs, Options, o =>
        {
            float sum = 0; int db = dy.StorageOffset + o * dy.ColumnStride;
            for (int n = 0; n < dy.Rows; n++) sum += da[db + n];
            float value = alpha * sum;
            target[0, o] = beta == 0 ? value : beta * target[0, o] + value;
        });
    }
}
