using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.FullyConnected;

/// <summary>Ordered float32 oracle. k ascends for forward, output ascends for dX,
/// cases ascend for dW/db. Separate multiply/add; no activation or optimizer.
/// Original src/layer.cu:634-675; A:src/nvmatrix/nvmatrix.cu:189-236,652-657,896-905.</summary>
public sealed class ScalarFullyConnected(FcGeometry geometry)
{
    public FcGeometry Geometry { get; } = geometry;
    public static ScalarFullyConnected FromLayer(LayerDefinition layer, LayerShape shape) => new(FcGeometry.FromLayer(layer, shape));

    public void Forward(IReadOnlyList<FloatMatrix> inputs, IReadOnlyList<FloatMatrix> weights, FloatMatrix bias, FloatMatrix target)
    {
        Geometry.ForwardAll(inputs, weights, bias, target);
        for (int i = 0; i < inputs.Count; i++) ForwardInput(i, inputs[i], weights[i], bias, target, i == 0 ? 0 : 1);
    }

    /// <summary>Bias follows this product exactly when scaleTargets==0. Whole-layer Forward
    /// overwrites on input 0, then adds remaining inputs in declared order.</summary>
    public void ForwardInput(int input, FloatMatrix x, FloatMatrix w, FloatMatrix bias, FloatMatrix target, float scaleTargets = 0)
    {
        Geometry.Forward(input, x, w, bias, target);
        for (int o = 0; o < Geometry.Outputs; o++)
            for (int n = 0; n < x.Rows; n++)
            {
                float sum = 0;
                for (int k = 0; k < x.Cols; k++) sum += x[n, k] * w[k, o];
                target[n, o] = scaleTargets == 0 ? sum : scaleTargets * target[n, o] + sum;
                if (scaleTargets == 0) target[n, o] += bias[0, o];
            }
    }

    /// <summary>No case averaging. Caller supplies per-producer contribution scale (0 then 1
    /// in a single buffer); original per-device counts are src/layer.cu:287-295.</summary>
    public void BackwardInput(int input, FloatMatrix dy, FloatMatrix w, FloatMatrix target, float scaleTargets = 0)
    {
        Geometry.Input(input, dy, w, target);
        for (int k = 0; k < target.Cols; k++)
            for (int n = 0; n < dy.Rows; n++)
            {
                float sum = 0;
                for (int o = 0; o < Geometry.Outputs; o++) sum += dy[n, o] * w[k, o];
                target[n, k] = scaleTargets == 0 ? sum : scaleTargets * target[n, k] + sum;
            }
    }

    /// <summary>target=beta*target+alpha*(X^T*dY). beta=0 skips old target;
    /// GEMM alpha=0 skips operands (including NaN). Scale completed sum once.</summary>
    public void BackwardWeights(int input, FloatMatrix x, FloatMatrix dy, FloatMatrix target, float alpha = 1, float beta = 0)
    {
        Geometry.Weight(input, x, dy, target);
        for (int o = 0; o < Geometry.Outputs; o++)
            for (int k = 0; k < x.Cols; k++)
            {
                float sum = 0;
                if (alpha != 0)
                {
                    for (int n = 0; n < x.Rows; n++) sum += x[n, k] * dy[n, o];
                    sum *= alpha;
                }
                target[k, o] = beta == 0 ? sum : beta * target[k, o] + sum;
            }
    }

    /// <summary>target=beta*target+alpha*sum_cases(dY). Unlike GEMM, addSum still
    /// evaluates the reduction when alpha=0 (A:nvmatrix.cu:896-900).</summary>
    public void BackwardBiases(FloatMatrix dy, FloatMatrix target, float alpha = 1, float beta = 0)
    {
        Geometry.Bias(dy, target);
        for (int o = 0; o < Geometry.Outputs; o++)
        {
            float sum = 0;
            for (int n = 0; n < dy.Rows; n++) sum += dy[n, o];
            float value = alpha * sum;
            target[0, o] = beta == 0 ? value : beta * target[0, o] + value;
        }
    }
}
