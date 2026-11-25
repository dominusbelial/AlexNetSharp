namespace AlexNetSharp.Core.Tensors;

public static class LayoutBridge
{
    /// <summary>NCHW is flat [case][channel][y][x]. Always copies; values are preserved.</summary>
    public static ActivationTensor FromNchw(ReadOnlySpan<float> nchw, int cases, int channels, int height, int width)
    {
        var tensor = new ActivationTensor(channels, height, width, cases);
        if (nchw.Length != tensor.Layout.ElementCount)
            throw new ArgumentException("NCHW length does not match dimensions.", nameof(nchw));
        for (int n = 0; n < cases; n++)
            for (int row = 0; row < tensor.Layout.Rows; row++)
                tensor.Matrix[row, n] = nchw[n * tensor.Layout.Rows + row];
        return tensor;
    }

    public static float[] ToNchw(ActivationTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        var nchw = new float[tensor.Layout.ElementCount];
        for (int n = 0; n < tensor.Layout.Cases; n++)
            for (int row = 0; row < tensor.Layout.Rows; row++)
                nchw[n * tensor.Layout.Rows + row] = tensor.Matrix[row, n];
        return nchw;
    }

    /// <summary>Import one already cropped 224x224 RGB image, interleaved [y][x][R,G,B],
    /// into a selected data-layer case. Keeps byte intensity units (no /255).
    /// Optional mean is planar [channel][y][x], already cropped to 224x224.
    /// RGB/planar order: convdata_jpeg.py:34-41,95,222; mean subtraction: :207-210.
    /// Crop/resize/JPEG decode/noise are deliberately outside this helper.</summary>
    public static void ImportRgbCrop(ReadOnlySpan<byte> rgb, ActivationTensor target, int @case,
        ReadOnlySpan<float> planarMean = default)
    {
        ArgumentNullException.ThrowIfNull(target);
        if (target.Layout.Channels != 3 || target.Layout.Height != 224 || target.Layout.Width != 224)
            throw new ArgumentException("RGB crop target must have 3 channels and a 224x224 plane.", nameof(target));
        FloatMatrix.CheckIndex(@case, target.Layout.Cases, nameof(@case));
        if (rgb.Length != 3 * 224 * 224)
            throw new ArgumentException("RGB crop must contain exactly 224x224x3 bytes.", nameof(rgb));
        if (!planarMean.IsEmpty && planarMean.Length != rgb.Length)
            throw new ArgumentException("Mean must be empty or contain 224x224x3 planar values.", nameof(planarMean));
        for (int channel = 0; channel < 3; channel++)
            for (int pixel = 0; pixel < 224 * 224; pixel++)
            {
                int row = channel * 224 * 224 + pixel;
                target.Matrix[row, @case] = rgb[3 * pixel + channel] - (planarMean.IsEmpty ? 0 : planarMean[row]);
            }
    }
}
