using AlexNetSharp.Core.Layers.Softmax;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Cost;

/// <summary>Parallel per-case cost kernels; final case sums run serially in the
/// same ascending order as the scalar reference. No shared reduction updates.</summary>
public static class OptimizedLogregCost
{
    public static LogregCostResult Forward(FloatMatrix labels, FloatMatrix probabilities, int topK = 1, int maxDegreeOfParallelism = -1)
    {
        LogregContract.Validate(labels, probabilities, topK); SoftmaxContract.Threads(maxDegreeOfParallelism);
        int cases = probabilities.Cols, classes = probabilities.Rows;
        var p = probabilities.BackingArray; var logs = new float[cases]; var correct = new float[cases]; var top = new float[cases];
        Parallel.For(0, cases, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, n =>
        {
            int index = probabilities.StorageOffset + n * probabilities.ColumnStride;
            float max = p[index], labelp = p[index + (int)labels[0, n] * probabilities.RowStride];
            for (int c = 1; c < classes; c++) { float v = p[index + c * probabilities.RowStride]; max = v > max ? v : max; }
            logs[n] = MathF.Log(labelp);
            int bigger = 0, equal = 0;
            for (int c = 0; c < classes; c++) { float v = p[index + c * probabilities.RowStride]; if (v > labelp) bigger++; if (v == labelp) equal++; }
            correct[n] = labelp != max ? 0 : 1f / equal;
            int slots = topK - bigger;
            top[n] = topK == 1 ? correct[n] : slots <= 0 ? 0 : equal <= slots ? 1 : (float)slots / equal;
        });
        float sumLog = 0, sumCorrect = 0, sumTop = 0;
        for (int n = 0; n < cases; n++) { sumLog += logs[n]; sumCorrect += correct[n]; sumTop += top[n]; }
        return new(-sumLog, cases - (double)sumCorrect, cases - (double)sumTop,
            Array.AsReadOnly(logs), Array.AsReadOnly(correct), Array.AsReadOnly(top));
    }
    public static void Backward(FloatMatrix labels, FloatMatrix probabilities, FloatMatrix target, float coefficient = 1, float scaleTargets = 0, int maxDegreeOfParallelism = -1)
    {
        LogregContract.Backward(labels, probabilities, target); SoftmaxContract.Threads(maxDegreeOfParallelism);
        var p = probabilities.BackingArray; var dx = target.BackingArray;
        Parallel.For(0, target.Cols, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, n =>
        {
            int label = (int)labels[0, n], pi = probabilities.StorageOffset + n * probabilities.ColumnStride, ti = target.StorageOffset + n * target.ColumnStride;
            for (int c = 0; c < target.Rows; c++)
            {
                float v = coefficient * (label == c ? 1f : 0f);
                v /= p[pi + c * probabilities.RowStride];
                int index = ti + c * target.RowStride;
                dx[index] = scaleTargets == 1 ? dx[index] + v : v;
            }
        });
    }
}
