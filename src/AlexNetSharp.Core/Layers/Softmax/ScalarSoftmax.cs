using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Softmax;

/// <summary>Native (cases,classes) F-order, including strided views. Forward:
/// first-class-seeded max, exp in ascending class order, first-class-seeded sum,
/// then divide. A:nvmatrix_kernels.cuh:544-556 and src/layer.cu:1020-1026.
/// MathF.Exp replaces CUDA __expf. General backward intentionally retains the
/// original O(classes²) reduction and rounding (layer_kernels.cu:486-490).
/// Fused backward is a negative loss gradient, with no batch division.
/// Softmax accumulation adds only when scaleTargets==1 (layer.cu:1038-1040).</summary>
public static class ScalarSoftmax
{
    public static void BackwardSelected(SoftmaxBackwardRoute route, FloatMatrix probabilities, FloatMatrix target,
        FloatMatrix? upstream = null, FloatMatrix? labels = null, float scaleTargets = 0)
    {
        if (route.FuseLogreg) BackwardLogreg(labels ?? throw new ArgumentNullException(nameof(labels)), probabilities, target, route.Coefficient, scaleTargets);
        else Backward(probabilities, upstream ?? throw new ArgumentNullException(nameof(upstream)), target, scaleTargets);
    }

    public static void Forward(FloatMatrix logits, FloatMatrix target)
    {
        SoftmaxContract.Same(target, logits);
        for (int n = 0; n < logits.Rows; n++)
        {
            float max = logits[n, 0];
            for (int c = 1; c < logits.Cols; c++) { float x = logits[n, c]; max = x > max ? x : max; }
            for (int c = 0; c < logits.Cols; c++) target[n, c] = MathF.Exp(logits[n, c] - max);
            float sum = target[n, 0];
            for (int c = 1; c < logits.Cols; c++) sum += target[n, c];
            for (int c = 0; c < logits.Cols; c++) target[n, c] /= sum;
        }
    }
    public static void Backward(FloatMatrix probabilities, FloatMatrix upstream, FloatMatrix target, float scaleTargets = 0)
    {
        SoftmaxContract.Same(target, probabilities, upstream);
        for (int n = 0; n < target.Rows; n++) for (int c = 0; c < target.Cols; c++)
        {
            float v = 0;
            for (int j = 0; j < target.Cols; j++) v += upstream[n, j] * ((j == c ? 1f : 0f) - probabilities[n, j]);
            v *= probabilities[n, c];
            target[n, c] = scaleTargets == 1 ? target[n, c] + v : v;
        }
    }
    public static void BackwardLogreg(FloatMatrix labels, FloatMatrix probabilities, FloatMatrix target, float coefficient = 1, float scaleTargets = 0)
    {
        SoftmaxContract.Same(target, probabilities);
        SoftmaxContract.Labels(labels, target.Rows, target.Cols); SoftmaxContract.Disjoint(target, labels);
        for (int n = 0; n < target.Rows; n++) for (int c = 0; c < target.Cols; c++)
        {
            float v = coefficient * (((int)labels[0, n] == c ? 1f : 0f) - probabilities[n, c]);
            target[n, c] = scaleTargets == 1 ? target[n, c] + v : v;
        }
    }
}
