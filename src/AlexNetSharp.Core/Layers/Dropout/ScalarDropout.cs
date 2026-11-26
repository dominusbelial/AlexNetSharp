using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Dropout;

/// <summary>src/layer.cu:1121-1153. Training multiplies by a saved unscaled
/// mask. All other modes, including enable=false training, multiply by keep.
/// Backward reuses the most recent forward's mode/mask; nonzero scaleTargets
/// adds (it does not multiply the old target). Exact in-place maps are supported.</summary>
public sealed class ScalarDropout(float keep, bool enable, IDropoutMaskSource maskSource)
{
    private readonly DropoutState _state = new(keep, enable, maskSource ?? throw new ArgumentNullException(nameof(maskSource)));
    public void Forward(FloatMatrix input, FloatMatrix target, DropoutMode mode = DropoutMode.Train)
    {
        _state.Prepare(input, target, mode, 1);
        for (int f = 0; f < input.Rows; f++) for (int n = 0; n < input.Cols; n++)
            target[f, n] = input[f, n] * (_state.Masked ? _state.Mask![f, n] : _state.Keep);
    }
    public void Backward(FloatMatrix upstream, FloatMatrix target, float scaleTargets = 0)
    {
        _state.Backward(upstream, target);
        for (int f = 0; f < target.Rows; f++) for (int n = 0; n < target.Cols; n++)
        {
            float v = upstream[f, n] * (_state.Masked ? _state.Mask![f, n] : _state.Keep);
            target[f, n] = scaleTargets != 0 ? (_state.Masked ? v + target[f, n] : target[f, n] + v) : v;
        }
    }
}
