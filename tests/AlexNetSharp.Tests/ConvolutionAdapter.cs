using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Layers.Conv;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

// Run the unchanged ANX-5 assertions against both engines without changing the oracle.
internal sealed class ConvolutionAdapter
{
    public static bool UseOptimized { get; set; }
    private readonly ScalarConvolution scalar;
    private readonly OptimizedConvolution? optimized;
    public int Filters => scalar.Filters;
    public int ModulesX => scalar.ModulesX;
    public int Modules => scalar.Modules;
    public IReadOnlyList<ConvInputGeometry> Inputs => scalar.Inputs;
    public ConvolutionAdapter(string name, IReadOnlyList<ConvInputGeometry> inputs, int filters,
        int partialSum = 0, bool sharedBiases = true, int threads = -1)
    {
        if (UseOptimized) optimized = new(name, inputs, filters, partialSum, sharedBiases, threads);
        scalar = new(name, inputs, filters, partialSum, sharedBiases);
    }
    public static ConvolutionAdapter FromLayer(LayerDefinition layer, LayerShape shape)
    {
        var s = ScalarConvolution.FromLayer(layer, shape);
        return new(s.Name, s.Inputs, s.Filters, s.PartialSum, s.SharedBiases);
    }
    public void Forward(IReadOnlyList<FloatMatrix> images, IReadOnlyList<FloatMatrix> weights, FloatMatrix biases, FloatMatrix target)
    { if (optimized is {} o) o.Forward(images, weights, biases, target); else scalar.Forward(images, weights, biases, target); }
    public void ForwardInput(int input, FloatMatrix images, FloatMatrix weights, FloatMatrix biases, FloatMatrix target, float scaleTargets = 0)
    { if (optimized is {} o) o.ForwardInput(input, images, weights, biases, target, scaleTargets); else scalar.ForwardInput(input, images, weights, biases, target, scaleTargets); }
    public void BackwardInput(int input, FloatMatrix outputGradient, FloatMatrix weights, FloatMatrix target, float scaleTargets = 0)
    { if (optimized is {} o) o.BackwardInput(input, outputGradient, weights, target, scaleTargets); else scalar.BackwardInput(input, outputGradient, weights, target, scaleTargets); }
    public void BackwardWeights(int input, FloatMatrix images, FloatMatrix outputGradient, FloatMatrix target,
        float scaleTargets = 0, bool gradientCheck = false, int? normalizationCases = null)
    { if (optimized is {} o) o.BackwardWeights(input, images, outputGradient, target, scaleTargets, gradientCheck, normalizationCases); else scalar.BackwardWeights(input, images, outputGradient, target, scaleTargets, gradientCheck, normalizationCases); }
    public void BackwardBiases(FloatMatrix outputGradient, FloatMatrix target,
        float scaleTargets = 0, bool gradientCheck = false, int? normalizationCases = null)
    { if (optimized is {} o) o.BackwardBiases(outputGradient, target, scaleTargets, gradientCheck, normalizationCases); else scalar.BackwardBiases(outputGradient, target, scaleTargets, gradientCheck, normalizationCases); }
}
