using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;

namespace AlexNetSharp.Tests;

internal static class ShapeTests
{
    internal static NetworkDefinition Parse145(string origRoot) => NetworkConfigParser.ParseFiles(
        Path.Combine(origRoot, "layers", "layers-145.cfg"),
        Path.Combine(origRoot, "layers", "layer-params-145.cfg"));

    public static int RunAll(string repoRoot, string origRoot)
    {
        int count = 0;
        var net = Parse145(origRoot);
        var saved = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("fr-FR");
            var fixture = JsonNode.Parse(File.ReadAllText(Path.Combine(repoRoot, "tests", "fixtures", "layers-145.shapes.json")))!;
            foreach (var c in fixture["cases"]!.AsArray())
            {
                int inner = c!["inner_size"]!.GetValue<int>();
                var actual = ShapeInference.Infer(net, DataDimensions.Jpeg(inner));
                EqualJson(c["layers"], JsonSerializer.SerializeToNode(actual.Layers, ShapeJson.Options), $"full network {inner}");
                TestAssert.Equal(c["weight_parameter_count"]!.GetValue<long>(), actual.WeightParameterCount, "weights count");
                TestAssert.Equal(c["bias_parameter_count"]!.GetValue<long>(), actual.BiasParameterCount, "bias count");
                TestAssert.SequenceEqual(net.Layers.Select(l => l.Name).ToArray(), actual.Layers.Select(l => l.Name).ToArray(), "parser order including detached neurons");
                count++;
            }
        }
        finally { CultureInfo.CurrentCulture = saved; }

        var standard = ShapeInference.Infer(net); // default inner_size must be 224
        TestAssert.Equal(150528, standard["data"].Outputs, "default data dimensions");
        TestAssert.Equal(55, standard["conv1a"].ModulesX, "hand conv1 width");
        TestAssert.Equal(true, standard["conv1a"].PartialBoundaryWindow[0], "224 conv1 partial");
        TestAssert.Equal(27, standard["pool1a"].OutputsX, "hand pool1 width");
        TestAssert.Equal(93312, standard["conv2a"].Outputs, "hand conv2 outputs");
        TestAssert.Equal(13, standard["pool2a"].OutputsX, "hand pool2 width");
        TestAssert.SequenceEqual([21632, 21632], standard["conv3a"].NumInputs, "hand conv3 two inputs");
        TestAssert.Equal(21632, standard["conv5a"].Outputs, "hand conv5 outputs");
        TestAssert.Equal(4608, standard["pool3a"].Outputs, "hand pool3 outputs");
        TestAssert.SequenceEqual([4608, 4608], standard["fc2048a"].NumInputs, "hand FC1 inputs");
        TestAssert.SequenceEqual(["hs1b", "hs1a"], standard["fc2048bb"].Inputs, "hand reversed FC2 input order");
        TestAssert.Equal(1000, standard["fc1000"].Outputs, "hand classes");
        TestAssert.Equal(60954656L, standard.WeightParameterCount, "hand total weights");
        TestAssert.Equal(60965224L, standard.WeightParameterCount + standard.BiasParameterCount, "hand total parameters");
        TestAssert.Equal(0, standard["logprob"].Outputs, "cost produces no activation output");
        count++;

        var boundaryFixture = JsonNode.Parse(File.ReadAllText(Path.Combine(repoRoot, "tests", "fixtures", "shape-cases.json")))!;
        foreach (var c in boundaryFixture["cases"]!.AsArray())
        {
            string id = c!["id"]!.GetValue<string>();
            var data = new DataDimensions(c["data_dimensions"]!.AsObject().ToDictionary(
                kv => int.Parse(kv.Key, CultureInfo.InvariantCulture), kv => kv.Value!.GetValue<int>()), c["classes"]!.GetValue<int>());
            NetworkShapes Run() => ShapeInference.Infer(NetworkConfigParser.Parse(
                c["layer_cfg"]!.GetValue<string>(), c["param_cfg"]!.GetValue<string>()), data);
            if (c["error"] is { } error)
            {
                string expected = error.GetValue<string>();
                var ex = TestAssert.ThrowsLayerError(() => Run(), expected, id);
                TestAssert.Equal(expected, ex.Message, id + " exact original message");
            }
            else
            {
                var shape = JsonSerializer.SerializeToNode(Run()["target"], ShapeJson.Options)!;
                foreach (var field in c["expected"]!.AsObject())
                    EqualJson(field.Value, shape[field.Key], $"{id}.{field.Key}");
            }
            count++;
        }

        // Ledger enforcement must fail for a missing field and a changed value, including neurons.
        var ledger = JsonNode.Parse(File.ReadAllText(Path.Combine(repoRoot, "docs", "spec", "ledger.json")))!;
        foreach (string collection in new[] { "sections", "detached_neuron_layers" })
        {
            var entry = ledger[collection]!.AsArray()[0]!["shape"]!["inference_by_inner_size"]!["224"]!.AsObject();
            var original = entry["outputs"]!.DeepClone();
            entry.Remove("outputs");
            var errors = new List<string>();
            ShapeSpecLinkChecks.Verify(net, ledger, errors);
            TestAssert.True(errors.Any(e => e.Contains("outputs", StringComparison.Ordinal)), "missing ledger output detected");
            entry["outputs"] = -1;
            errors.Clear();
            ShapeSpecLinkChecks.Verify(net, ledger, errors);
            TestAssert.True(errors.Any(e => e.Contains("outputs", StringComparison.Ordinal)), "wrong ledger output detected");
            entry["outputs"] = original;
            count += 2;
        }
        return count;
    }

    private static void EqualJson(JsonNode? expected, JsonNode? actual, string label)
    {
        if (!JsonNode.DeepEquals(expected, actual))
            throw new TestFailureException($"{label}: expected {expected}, got {actual}.");
    }
}
