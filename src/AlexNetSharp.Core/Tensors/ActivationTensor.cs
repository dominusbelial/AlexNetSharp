namespace AlexNetSharp.Core.Tensors;

/// <summary>Image activations in canonical feature-by-case row-major storage.
/// Matrix is always untransposed; explicitly request TransposeView for a _trans layer.</summary>
public sealed class ActivationTensor
{
    public ActivationLayout Layout { get; }
    public FloatMatrix Matrix { get; }

    public ActivationTensor(int channels, int height, int width, int cases)
    {
        Layout = new(channels, height, width, cases);
        Matrix = new(Layout.Rows, cases);
    }

    public float this[int channel, int y, int x, int @case]
    {
        get => Matrix[Layout.GetRow(channel, y, x), CheckCase(@case)];
        set => Matrix[Layout.GetRow(channel, y, x), CheckCase(@case)] = value;
    }

    /// <summary>A strided (features,1) shared view, without copying.</summary>
    public FloatMatrix CaseView(int @case) => Matrix.Slice(0, Layout.Rows, CheckCase(@case), 1);

    /// <summary>A shared (height,width) plane for a single case and channel.</summary>
    public FloatMatrix ChannelPlaneView(int channel, int @case) => new(
        Matrix.BackingArray,
        Layout.GetFlatOffset(channel, 0, 0, @case), Layout.Height, Layout.Width,
        checked(Layout.Width * Layout.Cases), Layout.Cases, MatrixLayout.RowMajor);

    private int CheckCase(int @case)
    {
        FloatMatrix.CheckIndex(@case, Layout.Cases, nameof(@case));
        return @case;
    }
}
