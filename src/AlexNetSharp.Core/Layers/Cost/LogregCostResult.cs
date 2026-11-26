using AlexNetSharp.Core.Layers.Softmax;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Cost;

/// <summary>Unweighted, unaveraged public cost triple, plus private-scratch analogues
/// useful for verification. Logs and sums are float32 before widening the totals.
/// coeff affects backward only (src/layer.cu:1833-1836,1855).</summary>
public sealed record LogregCostResult(double NegativeLogLikelihood, double Top1Error, double TopKError,
    IReadOnlyList<float> LabelLogProbabilities, IReadOnlyList<float> Top1Correct, IReadOnlyList<float> TopKCorrect);

internal static class LogregContract
{
    internal static void Validate(FloatMatrix labels, FloatMatrix probabilities, int topK)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        if (probabilities.IsTrans || probabilities.Rows <= 0 || probabilities.Cols <= 0)
            throw new ArgumentException("Logreg requires canonical C-order (classes,cases) probabilities.");
        SoftmaxContract.Labels(labels, probabilities.Cols, probabilities.Rows);
        if (topK < 1) throw new ArgumentOutOfRangeException(nameof(topK));
    }
    internal static void Backward(FloatMatrix labels, FloatMatrix probabilities, FloatMatrix target)
    {
        Validate(labels, probabilities, 1);
        ArgumentNullException.ThrowIfNull(target);
        if (target.IsTrans || target.Rows != probabilities.Rows || target.Cols != probabilities.Cols)
            throw new ArgumentException("Logreg gradient dimensions/layout disagree.");
        SoftmaxContract.Disjoint(target, labels); SoftmaxContract.Disjoint(target, probabilities);
    }
}
