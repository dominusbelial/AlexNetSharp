using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Softmax;

/// <summary>Independent flat-array kernels with disjoint case ownership. No
/// cross-case reductions, FMA or algebraic reassociation of the general gradient.</summary>
public static class OptimizedSoftmax
{
    public static void BackwardSelected(SoftmaxBackwardRoute route, FloatMatrix probabilities, FloatMatrix target,
        FloatMatrix? upstream = null, FloatMatrix? labels = null, float scaleTargets = 0, int maxDegreeOfParallelism = -1)
    {
        if (route.FuseLogreg) BackwardLogreg(labels ?? throw new ArgumentNullException(nameof(labels)), probabilities, target, route.Coefficient, scaleTargets, maxDegreeOfParallelism);
        else Backward(probabilities, upstream ?? throw new ArgumentNullException(nameof(upstream)), target, scaleTargets, maxDegreeOfParallelism);
    }

    public static void Forward(FloatMatrix logits, FloatMatrix target, int maxDegreeOfParallelism = -1)
    {
        SoftmaxContract.Same(target, logits); SoftmaxContract.Threads(maxDegreeOfParallelism);
        var x = logits.BackingArray; var y = target.BackingArray;
        Parallel.For(0, logits.Rows, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, n =>
        {
            int xi = logits.StorageOffset + n * logits.RowStride, yi = target.StorageOffset + n * target.RowStride;
            float max = x[xi];
            for (int c = 1; c < logits.Cols; c++) { float v = x[xi + c * logits.ColumnStride]; max = v > max ? v : max; }
            for (int c = 0; c < logits.Cols; c++) y[yi + c * target.ColumnStride] = MathF.Exp(x[xi + c * logits.ColumnStride] - max);
            float sum = y[yi];
            for (int c = 1; c < logits.Cols; c++) sum += y[yi + c * target.ColumnStride];
            for (int c = 0; c < logits.Cols; c++) y[yi + c * target.ColumnStride] /= sum;
        });
    }
    public static void Backward(FloatMatrix probabilities, FloatMatrix upstream, FloatMatrix target, float scaleTargets = 0, int maxDegreeOfParallelism = -1)
    {
        SoftmaxContract.Same(target, probabilities, upstream); SoftmaxContract.Threads(maxDegreeOfParallelism);
        var p = probabilities.BackingArray; var dy = upstream.BackingArray; var dx = target.BackingArray;
        Parallel.For(0, target.Rows, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, n =>
        {
            int pi = probabilities.StorageOffset + n * probabilities.RowStride;
            int ui = upstream.StorageOffset + n * upstream.RowStride, ti = target.StorageOffset + n * target.RowStride;
            for (int c = 0; c < target.Cols; c++)
            {
                float v = 0;
                for (int j = 0; j < target.Cols; j++) v += dy[ui + j * upstream.ColumnStride] * ((j == c ? 1f : 0f) - p[pi + j * probabilities.ColumnStride]);
                v *= p[pi + c * probabilities.ColumnStride];
                int index = ti + c * target.ColumnStride;
                dx[index] = scaleTargets == 1 ? dx[index] + v : v;
            }
        });
    }
    public static void BackwardLogreg(FloatMatrix labels, FloatMatrix probabilities, FloatMatrix target, float coefficient = 1, float scaleTargets = 0, int maxDegreeOfParallelism = -1)
    {
        SoftmaxContract.Same(target, probabilities); SoftmaxContract.Threads(maxDegreeOfParallelism);
        SoftmaxContract.Labels(labels, target.Rows, target.Cols); SoftmaxContract.Disjoint(target, labels);
        var p = probabilities.BackingArray; var dx = target.BackingArray;
        Parallel.For(0, target.Rows, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, n =>
        {
            int label = (int)labels[0, n], pi = probabilities.StorageOffset + n * probabilities.RowStride, ti = target.StorageOffset + n * target.RowStride;
            for (int c = 0; c < target.Cols; c++)
            {
                float v = coefficient * ((label == c ? 1f : 0f) - p[pi + c * probabilities.ColumnStride]);
                int index = ti + c * target.ColumnStride;
                dx[index] = scaleTargets == 1 ? dx[index] + v : v;
            }
        });
    }
}
