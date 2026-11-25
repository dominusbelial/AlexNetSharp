using System.Text.Json;
using System.Text.Json.Serialization;

namespace AlexNetSharp.Core.Shapes;

/// <summary>Provider dimensions per dataIdx and dataset class count (layer.py:1099,1268).
/// These are per case, without a batch axis.</summary>
public sealed record DataDimensions(IReadOnlyDictionary<int, int> Dimensions, int Classes = 1000)
{
    /// <summary>224 is the paper default, not a recovered launch setting (OQ-LEDGER-1).
    /// innerSize = img_size - 2 * crop_border (convdata_jpeg.py:73,99-102).</summary>
    public static DataDimensions Jpeg(int innerSize = 224, int classes = 1000)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(innerSize);
        return new(new Dictionary<int, int> { [0] = checked(innerSize * innerSize * 3), [1] = 1 }, classes);
    }
}

/// <summary>Original parser fields: imgSize/channels describe image INPUTS, in input order.
/// Empty arrays/null mean inapplicable (including image metadata on data/FC/neuron layers).
/// PartialBoundaryWindow is per input: last window exceeds the right/bottom padded bound;
/// ordinary configured convolution padding is not itself a partial window.</summary>
public sealed record LayerShape(
    string Name, string Type, IReadOnlyList<string> Inputs, IReadOnlyList<int> NumInputs,
    IReadOnlyList<int> ImgSize, IReadOnlyList<int> Channels,
    int? ModulesX, int? Modules, int? OutputsX, int Outputs,
    IReadOnlyList<int> FilterChannels, IReadOnlyList<int> FilterPixels,
    IReadOnlyList<int[]> WeightsShapes, IReadOnlyList<int> BiasesShape,
    IReadOnlyList<bool> PartialBoundaryWindow);

public sealed class NetworkShapes(IReadOnlyList<LayerShape> layers)
{
    public IReadOnlyList<LayerShape> Layers { get; } = layers;
    private readonly Dictionary<string, LayerShape> _byName = layers.ToDictionary(l => l.Name, StringComparer.Ordinal);
    public LayerShape this[string name] => _byName[name];

    // Counts matrix slots; layers-145 has no shared weight matrices.
    public long WeightParameterCount => Layers.Sum(l => l.WeightsShapes.Sum(w => (long)w[0] * w[1]));
    public long BiasParameterCount => Layers.Sum(l => l.BiasesShape.Count == 0 ? 0L : (long)l.BiasesShape[0] * l.BiasesShape[1]);
}

public static class ShapeJson
{
    public static JsonSerializerOptions Options { get; } = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
    };
}
