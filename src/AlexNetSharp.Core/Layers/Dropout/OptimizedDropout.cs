using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Dropout;

/// <summary>Independent flat-stride arithmetic; only cases run in parallel.
/// Same saved-state contract and float operations as ScalarDropout.</summary>
public sealed class OptimizedDropout
{
    private readonly DropoutState _state;
    private readonly int _threads;
    public OptimizedDropout(float keep, bool enable, IDropoutMaskSource maskSource, int maxDegreeOfParallelism = -1)
    {
        if (maxDegreeOfParallelism == 0 || maxDegreeOfParallelism < -1) throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism));
        _state = new(keep, enable, maskSource ?? throw new ArgumentNullException(nameof(maskSource))); _threads = maxDegreeOfParallelism;
    }
    public void Forward(FloatMatrix input, FloatMatrix target, DropoutMode mode = DropoutMode.Train)
    {
        _state.Prepare(input, target, mode, _threads);
        Execute(input, target, false, 0);
    }
    public void Backward(FloatMatrix upstream, FloatMatrix target, float scaleTargets = 0)
    {
        _state.Backward(upstream, target);
        Execute(upstream, target, true, scaleTargets);
    }
    private void Execute(FloatMatrix source, FloatMatrix target, bool backward, float scale)
    {
        var x = source.BackingArray; var y = target.BackingArray; var mask = _state.Mask;
        Parallel.For(0, source.Cols, new ParallelOptions { MaxDegreeOfParallelism = _threads }, n =>
        {
            int xi = source.StorageOffset + n * source.ColumnStride, yi = target.StorageOffset + n * target.ColumnStride;
            for (int f = 0; f < source.Rows; f++, xi += source.RowStride, yi += target.RowStride)
            {
                float multiplier = _state.Masked ? mask!.BackingArray[mask.StorageOffset + f * mask.RowStride + n * mask.ColumnStride] : _state.Keep;
                float value = x[xi] * multiplier;
                y[yi] = backward && scale != 0 ? (_state.Masked ? value + y[yi] : y[yi] + value) : value;
            }
        });
    }
}
