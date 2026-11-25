namespace AlexNetSharp.Core.Tensors;

public readonly record struct ActivationIndex(int Channel, int Y, int X, int Case);

/// <summary>Canonical (channel,y,x,case) index map: case is fastest in memory,
/// then x, y, channel (Candidate A filter_acts.cu:1209-1213; conv_util.cuh:121-122).</summary>
public sealed class ActivationLayout
{
    public int Channels { get; }
    public int Height { get; }
    public int Width { get; }
    public int Cases { get; }
    public int PixelsPerChannel { get; }
    public int Rows { get; }
    public int ElementCount { get; }

    public ActivationLayout(int channels, int height, int width, int cases)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cases);
        Channels = channels;
        Height = height;
        Width = width;
        Cases = cases;
        PixelsPerChannel = checked(height * width);
        Rows = checked(channels * PixelsPerChannel);
        ElementCount = checked(Rows * cases);
    }

    public int GetRow(int channel, int y, int x)
    {
        FloatMatrix.CheckIndex(channel, Channels, nameof(channel));
        FloatMatrix.CheckIndex(y, Height, nameof(y));
        FloatMatrix.CheckIndex(x, Width, nameof(x));
        return (channel * Height + y) * Width + x;
    }

    public (int Row, int Col) ToMatrixIndex(int channel, int y, int x, int @case)
    {
        FloatMatrix.CheckIndex(@case, Cases, nameof(@case));
        return (GetRow(channel, y, x), @case);
    }

    public ActivationIndex FromMatrixIndex(int row, int col)
    {
        FloatMatrix.CheckIndex(row, Rows, nameof(row));
        FloatMatrix.CheckIndex(col, Cases, nameof(col));
        int pixel = row % PixelsPerChannel;
        return new(row / PixelsPerChannel, pixel / Width, pixel % Width, col);
    }

    public int GetFlatOffset(int channel, int y, int x, int @case)
    {
        var (row, col) = ToMatrixIndex(channel, y, x, @case);
        return row * Cases + col;
    }

    public ActivationIndex FromFlatOffset(int offset)
    {
        FloatMatrix.CheckIndex(offset, ElementCount, nameof(offset));
        return FromMatrixIndex(offset / Cases, offset % Cases);
    }
}
