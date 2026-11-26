using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Softmax;

/// <summary>The exact consumer test in src/layer.cu:1034. Disabling the softmax
/// flag forces its general backward, but the cost's skip test (1853) ignores that
/// flag. A caller forcing general backward must explicitly supply probability dy.</summary>
public static class SoftmaxFusion
{
    public static bool Select(bool doLogregGradient, int consumerCount, string consumerType, int softmaxDevice, int consumerDevice) =>
        doLogregGradient && consumerCount == 1 && consumerType == "cost.logreg" && softmaxDevice == consumerDevice;

    public static bool Select(NetworkDefinition network, string softmaxName, bool doLogregGradient = true)
    {
        var layer = network[softmaxName];
        if (layer.Type != "softmax") throw new ArgumentException("Softmax layer required.");
        var consumers = network.Layers.Where(l => l.Inputs.Contains(softmaxName)).ToArray();
        return consumers.Length == 1 && Select(doLogregGradient, consumers.Length, consumers[0].Type,
            layer.TypedParams.Gpu, consumers[0].TypedParams.Gpu);
    }

    public static bool CostComputesProbabilityGradient(int producerConsumerCount, string producerType, int producerDevice, int costDevice) =>
        producerConsumerCount > 1 || producerType != "softmax" || producerDevice != costDevice;
}

internal static class SoftmaxContract
{
    internal static void Native(FloatMatrix m)
    {
        ArgumentNullException.ThrowIfNull(m);
        if (!m.IsTrans || m.Rows <= 0 || m.Cols <= 0) throw new ArgumentException("Softmax requires native F-order (cases,classes) matrices.");
    }
    internal static void Same(FloatMatrix target, params FloatMatrix[] sources)
    {
        Native(target);
        foreach (var s in sources)
        {
            Native(s);
            if (target.Rows != s.Rows || target.Cols != s.Cols) throw new ArgumentException("Softmax dimensions disagree.");
            Disjoint(target, s);
        }
    }
    internal static void Disjoint(FloatMatrix target, FloatMatrix source)
    {
        if (ReferenceEquals(target.BackingArray, source.BackingArray)) throw new ArgumentException("Softmax/cost target must not share operand storage.");
    }
    internal static void Labels(FloatMatrix labels, int cases, int classes)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (labels.IsTrans || labels.Rows != 1 || labels.Cols != cases) throw new ArgumentException("Labels require C-order (1,cases).");
        for (int n = 0; n < cases; n++)
        {
            float value = labels[0, n];
            // The CUDA code int(labels[n]) truncates toward zero. Reject unsafe
            // nonfinite / out-of-range indices before any output is modified.
            if (!float.IsFinite(value) || value <= -1 || (double)value >= classes)
                throw new ArgumentException("Label index is outside the class range.");
        }
    }
    internal static void Threads(int threads)
    {
        if (threads == 0 || threads < -1) throw new ArgumentOutOfRangeException(nameof(threads));
    }
}

/// <summary>Small dispatch descriptor, not a graph executor. The consumer's
/// coefficient is captured from parsed definitions only for the fused route.</summary>
public sealed record SoftmaxBackwardRoute(bool FuseLogreg, float Coefficient = 1)
{
    public static SoftmaxBackwardRoute FromNetwork(NetworkDefinition network, string softmaxName, bool doLogregGradient = true)
    {
        bool fuse = SoftmaxFusion.Select(network, softmaxName, doLogregGradient);
        if (!fuse) return new(false);
        var consumer = network.Layers.Single(l => l.Inputs.Contains(softmaxName));
        return new(true, (float)((LogregCostLayerParams)consumer.TypedParams).Coeff);
    }
}
