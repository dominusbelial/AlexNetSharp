using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Pool;

/// <summary>Original layer.py:1139,1164: default start=0 and ceiling output width.
/// Explicit OutputsX overrides inference, including clipped/empty windows.</summary>
public sealed record PoolGeometry(int ImageSize, int Channels, int SizeX, int Stride,
    int Start = 0, int OutputsX = 0)
{
    public int OutputSize => OutputsX > 0 ? OutputsX : checked(1 + (int)Math.Ceiling(
        (ImageSize - (double)Start - SizeX) / Stride));
    public int InputRows => checked(Channels * ImageSize * ImageSize);
    public int OutputRows => checked(Channels * OutputSize * OutputSize);
}

internal static class PoolContract
{
    internal static void Geometry(string name, PoolGeometry g, string pool)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(g);
        if (pool != "max" && pool != "avg")
            throw new NotSupportedException($"Layer '{name}': pool={pool} is unsupported in this port.");
        if (g.ImageSize <= 0 || g.Channels <= 0 || g.SizeX <= 0 || g.SizeX > g.ImageSize ||
            g.Stride <= 0 || g.Stride > g.SizeX || g.OutputsX < 0 || g.OutputSize <= 0)
            throw new ArgumentException($"Layer '{name}': invalid pooling geometry.");
        _ = g.InputRows;
        _ = g.OutputRows;
        _ = checked(g.Start + checked((g.OutputSize - 1) * g.Stride) + g.SizeX);
    }

    internal static PoolGeometry FromLayer(LayerDefinition layer, LayerShape shape)
    {
        if (layer.TypedParams is not PoolLayerParams p || shape.Type != "pool" || layer.Name != shape.Name)
            throw new ArgumentException("Matching pooling definition and shape required.");
        var g = new PoolGeometry(shape.ImgSize[0], p.Channels, p.SizeX, p.Stride, p.Start, p.OutputsX);
        if (g.OutputSize != shape.OutputsX || g.OutputRows != shape.Outputs)
            throw new ArgumentException($"Layer '{layer.Name}': definition and shape disagree.");
        return g;
    }

    internal static void Matrix(string name, FloatMatrix m, int rows, int cases)
    {
        ArgumentNullException.ThrowIfNull(m);
        if (m.Layout != MatrixLayout.RowMajor || m.ColumnStride != 1 || m.Rows != rows || m.Cols != cases || cases <= 0)
            throw new ArgumentException($"Layer '{name}': expected canonical C-order ({rows},{cases}) matrix.");
    }

    internal static void Disjoint(string name, FloatMatrix target, params FloatMatrix[] sources)
    {
        if (sources.Any(s => ReferenceEquals(target.BackingArray, s.BackingArray)))
            throw new ArgumentException($"Layer '{name}': pooling target must not share operand storage.");
    }
}
