using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Norm;

/// <summary>Square image planes, channel/y/x rows, cases in contiguous columns.
/// CPU kernels allow tiny planes/channels below Size, unlike CUDA launch constraints.</summary>
public sealed record NormGeometry(int ImageSize, int Channels, int Size)
{
    public int Rows => checked(Channels * ImageSize * ImageSize);
}

internal static class NormContract
{
    internal static void Validate(string name, NormGeometry g, string kind)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(g);
        if (kind != "cmrnorm" && kind != "rnorm")
            throw new NotSupportedException($"Layer '{name}': {kind} is unsupported in this port.");
        if (g.ImageSize <= 0 || g.Channels <= 0 || g.Size <= 0)
            throw new ArgumentException($"Layer '{name}': invalid normalisation geometry.");
        _ = g.Rows;
        _ = checked(g.Size + Math.Max(g.Channels, g.ImageSize));
    }

    internal static (NormGeometry Geometry, float Scale, float Power, float MinDiv, bool Blocked) FromLayer(
        LayerDefinition layer, LayerShape shape)
    {
        if (layer.Name != shape.Name || layer.Type != shape.Type || shape.ImgSize.Count != 1)
            throw new ArgumentException($"Layer '{layer.Name}': matching normalisation definition and shape required.");
        var settings = layer.TypedParams switch
        {
            CmrNormLayerParams p when layer.Type == "cmrnorm" =>
                (new NormGeometry(shape.ImgSize[0], p.Channels, p.Size), (float)p.Scale, (float)p.Pow, (float)p.MinDiv, p.Blocked),
            RNormLayerParams p when layer.Type == "rnorm" =>
                (new NormGeometry(shape.ImgSize[0], p.Channels, p.Size), (float)p.Scale, (float)p.Pow, 1f, p.Blocked),
            _ => throw new ArgumentException($"Layer '{layer.Name}': normalisation parameters required.")
        };
        if (settings.Item1.Rows != shape.Outputs || shape.NumInputs.Count != 1 || shape.NumInputs[0] != shape.Outputs)
            throw new ArgumentException($"Layer '{layer.Name}': definition and shape disagree.");
        return settings;
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
            throw new ArgumentException($"Layer '{name}': normalisation target must not share operand storage.");
    }
}
