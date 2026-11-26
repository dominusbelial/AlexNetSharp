using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Cost;

/// <summary>src/layer_kernels.cu:96-119,387-445. Natural MathF.Log replaces
/// CUDA __logf, without clipping: log(0)=-infinity. Fractional ties: top1=1/E
/// if labelp=maxp; topk=clamp((k-B)/E,0,1), B=#greater, E=#equal to labelp.
/// Preserve the original comparisons even for NaN (E=0). Fixed class/case order.</summary>
public static class ScalarLogregCost
{
    public static LogregCostResult Forward(FloatMatrix labels, FloatMatrix probabilities, int topK = 1)
    {
        LogregContract.Validate(labels, probabilities, topK);
        int cases = probabilities.Cols, classes = probabilities.Rows;
        var logs = new float[cases]; var correct = new float[cases]; var top = new float[cases];
        for (int n = 0; n < cases; n++)
        {
            float labelp = probabilities[(int)labels[0, n], n], max = probabilities[0, n];
            for (int c = 1; c < classes; c++) { float p = probabilities[c, n]; max = p > max ? p : max; }
            logs[n] = MathF.Log(labelp);
            int bigger = 0, equal = 0;
            for (int c = 0; c < classes; c++) { float p = probabilities[c, n]; if (p > labelp) bigger++; if (p == labelp) equal++; }
            correct[n] = labelp != max ? 0 : 1f / equal;
            int slots = topK - bigger;
            top[n] = topK == 1 ? correct[n] : slots <= 0 ? 0 : equal <= slots ? 1 : (float)slots / equal;
        }
        float sumLog = 0, sumCorrect = 0, sumTop = 0;
        for (int n = 0; n < cases; n++) { sumLog += logs[n]; sumCorrect += correct[n]; sumTop += top[n]; }
        return new(-sumLog, cases - (double)sumCorrect, cases - (double)sumTop,
            Array.AsReadOnly(logs), Array.AsReadOnly(correct), Array.AsReadOnly(top));
    }
    /// <summary>Non-fused probability negative gradient coeff*indicator/p, for
    /// EVERY class (0/0 remains NaN). Cost normally skips this for the fused edge.</summary>
    public static void Backward(FloatMatrix labels, FloatMatrix probabilities, FloatMatrix target, float coefficient = 1, float scaleTargets = 0)
    {
        LogregContract.Backward(labels, probabilities, target);
        for (int n = 0; n < target.Cols; n++) for (int c = 0; c < target.Rows; c++)
        {
            float v = coefficient * ((int)labels[0, n] == c ? 1f : 0f);
            v /= probabilities[c, n];
            target[c, n] = scaleTargets == 1 ? target[c, n] + v : v;
        }
    }
}
