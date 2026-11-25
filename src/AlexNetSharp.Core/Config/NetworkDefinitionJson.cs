using System.Text;
using System.Text.Json;

namespace AlexNetSharp.Core.Config;

/// <summary>
/// Deterministic, culture-invariant JSON serializer for <see cref="NetworkDefinition"/>.
/// Always emits LF (<c>\n</c>) line endings and stable <see cref="StringComparer.Ordinal"/> key order.
/// </summary>
public static class NetworkDefinitionJson
{
    public static string Serialize(NetworkDefinition network)
    {
        ArgumentNullException.ThrowIfNull(network);

        using var ms = new MemoryStream();
        var options = new JsonWriterOptions
        {
            Indented = true,
            NewLine = "\n",
        };

        using (var writer = new Utf8JsonWriter(ms, options))
        {
            writer.WriteStartObject();
            writer.WriteNumber("schemaVersion", 1);
            writer.WriteNumber("layerCount", network.Layers.Count);
            writer.WriteNumber("declaredLayerCount", network.DeclaredLayers.Count);
            writer.WriteNumber("detachedNeuronLayerCount", network.DetachedNeuronLayers.Count);

            writer.WritePropertyName("layers");
            writer.WriteStartArray();
            foreach (var layer in network.Layers)
                WriteLayer(writer, layer);
            writer.WriteEndArray();

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(ms.ToArray()) + "\n";
    }

    private static void WriteLayer(Utf8JsonWriter writer, LayerDefinition layer)
    {
        writer.WriteStartObject();
        writer.WriteString("name", layer.Name);
        writer.WriteString("type", layer.Type);

        writer.WritePropertyName("declaredInputs");
        writer.WriteStartArray();
        foreach (var inp in layer.DeclaredInputs)
            writer.WriteStringValue(inp);
        writer.WriteEndArray();

        writer.WritePropertyName("inputs");
        writer.WriteStartArray();
        foreach (var inp in layer.Inputs)
            writer.WriteStringValue(inp);
        writer.WriteEndArray();

        if (layer.SourceLayer is not null)
            writer.WriteString("sourceLayer", layer.SourceLayer);
        else
            writer.WriteNull("sourceLayer");

        if (layer.DetachedNeuronLayer is not null)
            writer.WriteString("detachedNeuronLayer", layer.DetachedNeuronLayer);
        else
            writer.WriteNull("detachedNeuronLayer");

        writer.WriteBoolean("requiresParams", layer.RequiresParams);

        writer.WritePropertyName("parameters");
        writer.WriteStartObject();
        foreach (var kv in layer.Parameters.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            writer.WritePropertyName(kv.Key);
            WriteConfigValue(writer, kv.Value);
        }
        writer.WriteEndObject();

        writer.WriteEndObject();
    }

    private static void WriteConfigValue(Utf8JsonWriter writer, ConfigValue val)
    {
        switch (val.Kind)
        {
            case ConfigValueKind.Int:
                writer.WriteNumberValue(val.AsInt());
                break;
            case ConfigValueKind.Float:
                writer.WriteNumberValue(NormalizeZero(val.AsFloat()));
                break;
            case ConfigValueKind.Bool:
                writer.WriteBooleanValue(val.AsBool());
                break;
            case ConfigValueKind.String:
                writer.WriteStringValue(val.AsString());
                break;
            case ConfigValueKind.IntList:
                writer.WriteStartArray();
                foreach (var x in val.AsIntList())
                    writer.WriteNumberValue(x);
                writer.WriteEndArray();
                break;
            case ConfigValueKind.FloatList:
                writer.WriteStartArray();
                foreach (var x in val.AsFloatList())
                    writer.WriteNumberValue(NormalizeZero(x));
                writer.WriteEndArray();
                break;
            case ConfigValueKind.BoolList:
                writer.WriteStartArray();
                foreach (var x in val.AsBoolList())
                    writer.WriteBooleanValue(x);
                writer.WriteEndArray();
                break;
            case ConfigValueKind.StringList:
                writer.WriteStartArray();
                foreach (var x in val.AsStringList())
                    writer.WriteStringValue(x);
                writer.WriteEndArray();
                break;
        }
    }

    private static double NormalizeZero(double v) => v == 0.0 ? 0.0 : v;
}
