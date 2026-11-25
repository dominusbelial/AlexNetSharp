using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;

namespace AlexNetSharp.Tests;

internal static class ShapeSpecLinkChecks
{
    public static int Verify(NetworkDefinition network, JsonNode ledger, List<string> errors)
    {
        var byName = ledger["sections"]!.AsArray().Concat(ledger["detached_neuron_layers"]!.AsArray())
            .ToDictionary(n => n!["name"]!.GetValue<string>(), n => n!, StringComparer.Ordinal);
        int checkedFields = 0;
        foreach (int inner in new[] { 224, 227 })
        {
            foreach (var layer in ShapeInference.Infer(network, DataDimensions.Jpeg(inner)).Layers)
            {
                string prefix = $"shape-link['{layer.Name}'][{inner}]";
                if (!byName.TryGetValue(layer.Name, out var section))
                {
                    errors.Add($"{prefix}: missing ledger layer.");
                    continue;
                }
                var expected = section["shape"]?["inference_by_inner_size"]?[inner.ToString(CultureInfo.InvariantCulture)] as JsonObject;
                var actual = JsonSerializer.SerializeToNode(layer, ShapeJson.Options)!.AsObject();
                foreach (var field in actual)
                {
                    checkedFields++;
                    if (expected is null || !expected.TryGetPropertyValue(field.Key, out var value))
                        errors.Add($"{prefix}.{field.Key}: emitted field missing from ledger shape.");
                    else if (!JsonNode.DeepEquals(field.Value, value))
                        errors.Add($"{prefix}.{field.Key}: computed {field.Value} != ledger {value}.");
                }
                if (expected is not null)
                    foreach (var field in expected)
                        if (!actual.ContainsKey(field.Key)) errors.Add($"{prefix}.{field.Key}: ledger field not emitted.");

                // Keep the pre-existing ANX-1 numbers independently linked, too.
                if (inner == 224 && section["shape"] is { } old)
                {
                    if (old["outputs"] is { } n && n.GetValue<int>() != layer.Outputs)
                        errors.Add($"{prefix}.outputs: differs from original ANX-1 shape.outputs.");
                    if (old["weights_shapes"] is { } w && !JsonNode.DeepEquals(w, actual["weights_shapes"]))
                        errors.Add($"{prefix}.weights_shapes: differs from original ANX-1 weights.");
                    if (old["biases_shape"] is { } b && !JsonNode.DeepEquals(b, actual["biases_shape"]))
                        errors.Add($"{prefix}.biases_shape: differs from original ANX-1 biases.");
                }
            }
        }
        return checkedFields;
    }
}
