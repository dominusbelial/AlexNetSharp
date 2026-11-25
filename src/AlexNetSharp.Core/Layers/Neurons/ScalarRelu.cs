using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Neurons;

/// <summary>Original include/neuron.cuh:146-170: forward x&lt;0?0:x (preserves
/// negative zero/NaN); backward upstream*(savedOutput&gt;0). scaleTargets&gt;0 adds
/// the derived gradient to the target, otherwise overwrites (layer.cu:504).
/// Both layouts and strided views work; exact in-place mappings are supported.</summary>
public static class ScalarRelu
{
    public static void Forward(FloatMatrix input, FloatMatrix target)
    {
        ReluContract.Validate(target, input);
        for (int r = 0; r < input.Rows; r++)
            for (int n = 0; n < input.Cols; n++)
            {
                float x = input[r, n];
                target[r, n] = x < 0 ? 0 : x;
            }
    }
    public static void Backward(FloatMatrix output, FloatMatrix outputGradient, FloatMatrix target, float scaleTargets = 0)
    {
        ReluContract.Validate(target, output, outputGradient);
        for (int r = 0; r < target.Rows; r++)
            for (int n = 0; n < target.Cols; n++)
            {
                float dx = outputGradient[r, n] * (output[r, n] > 0 ? 1f : 0f);
                target[r, n] = scaleTargets > 0 ? dx + target[r, n] : dx;
            }
    }
}

internal static class ReluContract
{
    internal static void Validate(FloatMatrix target, params FloatMatrix[] sources)
    {
        ArgumentNullException.ThrowIfNull(target);
        foreach (var s in sources)
        {
            ArgumentNullException.ThrowIfNull(s);
            if (s.Rows != target.Rows || s.Cols != target.Cols)
                throw new ArgumentException("ReLU requires equal matrix dimensions.");
            if (ReferenceEquals(target.BackingArray, s.BackingArray) &&
                (target.StorageOffset != s.StorageOffset || target.RowStride != s.RowStride || target.ColumnStride != s.ColumnStride))
                throw new ArgumentException("ReLU rejects overlapping views with different index maps.");
        }
    }
}
