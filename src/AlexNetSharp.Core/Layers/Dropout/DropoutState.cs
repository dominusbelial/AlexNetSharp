using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Dropout;

public enum DropoutMode { Train, Evaluate, GradientCheck, FeatureGeneration, MultiviewTest }

// Shared validation/state, not shared scalar/optimized arithmetic.
internal sealed class DropoutState(float keep, bool enable, IDropoutMaskSource source)
{
    internal float Keep { get; } = keep;
    internal FloatMatrix? Mask { get; private set; }
    internal bool Masked { get; private set; }
    private bool _ready;
    private int _features, _cases;
    internal void Prepare(FloatMatrix input, FloatMatrix target, DropoutMode mode, int threads)
    {
        Validate(input, target);
        if (!Enum.IsDefined(mode)) throw new ArgumentOutOfRangeException(nameof(mode));
        // Invalid fills cannot leave an apparently usable saved forward.
        _ready = false;
        Masked = enable && mode == DropoutMode.Train;
        if (Masked)
        {
            if (Mask == null || Mask.Rows != input.Rows || Mask.Cols != input.Cols) Mask = new(input.Rows, input.Cols);
            source.Fill(Mask, Keep, threads);
            for (int f = 0; f < Mask.Rows; f++) for (int n = 0; n < Mask.Cols; n++)
                if (Mask[f, n] != 0 && Mask[f, n] != 1) throw new ArgumentException("Mask source must write binary values.");
        }
        _features = input.Rows; _cases = input.Cols;
        _ready = true;
    }
    internal void Backward(FloatMatrix dy, FloatMatrix target)
    {
        Validate(dy, target);
        if (!_ready || dy.Rows != _features || dy.Cols != _cases)
            throw new InvalidOperationException("Backward requires a matching completed forward.");
    }
    internal static void Validate(FloatMatrix source, FloatMatrix target)
    {
        ArgumentNullException.ThrowIfNull(source); ArgumentNullException.ThrowIfNull(target);
        if (source.IsTrans || target.IsTrans || source.Rows <= 0 || source.Cols <= 0 ||
            source.Rows != target.Rows || source.Cols != target.Cols)
            throw new ArgumentException("Dropout requires equal canonical C-order (features,cases) matrices.");
        if (ReferenceEquals(source.BackingArray, target.BackingArray) &&
            (source.StorageOffset != target.StorageOffset || source.RowStride != target.RowStride || source.ColumnStride != target.ColumnStride))
            throw new ArgumentException("Dropout rejects overlapping views with different index maps.");
    }
}
