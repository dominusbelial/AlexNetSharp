using System.Text.Json;
using AlexNetSharp.Core.Config;

namespace AlexNetSharp.Tests;

internal static class SpecLinkChecks
{
    private static readonly HashSet<(string Type, string Param)> ParserStoredParams =
    [
        ("conv", "padding"),
        ("cmrnorm", "scale"),
        ("rnorm", "scale"),
    ];

    public static int VerifyParserAgainstLedger(
        string origRoot,
        string ledgerPath,
        List<string> errors)
    {
        if (!File.Exists(ledgerPath))
            return 0;

        var layerCfgPath = Path.Combine(origRoot, "layers", "layers-145.cfg");
        var paramCfgPath = Path.Combine(origRoot, "layers", "layer-params-145.cfg");
        if (!File.Exists(layerCfgPath) || !File.Exists(paramCfgPath))
            return 0;

        var network = NetworkConfigParser.ParseFiles(layerCfgPath, paramCfgPath);
        using var doc = JsonDocument.Parse(File.ReadAllText(ledgerPath));
        var root = doc.RootElement;

        if (!root.TryGetProperty("sections", out var sectionsEl) || sectionsEl.ValueKind != JsonValueKind.Array)
            return 0;

        var ledgerByName = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var sec in sectionsEl.EnumerateArray())
        {
            if (sec.TryGetProperty("name", out var nEl) && nEl.ValueKind == JsonValueKind.String)
                ledgerByName[nEl.GetString()!] = sec;
        }

        int checkedParams = 0;
        foreach (var layer in network.DeclaredLayers)
        {
            if (!ledgerByName.TryGetValue(layer.Name, out var ledgerSec))
                continue; // Missing section already reported by VerifyLedger

            if (!ledgerSec.TryGetProperty("parameters", out var ledgerParams) ||
                ledgerParams.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            if (ledgerSec.TryGetProperty("effective_inputs", out var effIn) && effIn.ValueKind == JsonValueKind.Array)
            {
                var expectedEff = effIn.EnumerateArray().Select(x => x.GetString() ?? "").ToArray();
                if (!layer.Inputs.SequenceEqual(expectedEff, StringComparer.Ordinal))
                {
                    errors.Add(
                        $"spec-link['{layer.Name}']: parser effective inputs [{string.Join(",", layer.Inputs)}] != ledger [{string.Join(",", expectedEff)}].");
                }
            }

            var ledgerKeys = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in ledgerParams.EnumerateObject())
                ledgerKeys.Add(p.Name);

            foreach (var kv in layer.Parameters)
            {
                checkedParams++;
                if (!ledgerParams.TryGetProperty(kv.Key, out var pEntry))
                {
                    errors.Add(
                        $"spec-link['{layer.Name}']: parser emitted parameter '{kv.Key}' which is missing from ledger.sections['{layer.Name}'].parameters.");
                    continue;
                }

                var targetField = ParserStoredParams.Contains((layer.Type, kv.Key)) &&
                                  pEntry.TryGetProperty("parser_stored", out _)
                    ? "parser_stored"
                    : "value";

                if (pEntry.TryGetProperty(targetField, out var expectedVal))
                    VerifyValueMatchesJson(layer.Name, kv.Key, kv.Value, expectedVal, errors);
            }

            foreach (var lk in ledgerKeys.OrderBy(x => x, StringComparer.Ordinal))
            {
                if (!layer.Parameters.ContainsKey(lk))
                {
                    errors.Add(
                        $"spec-link['{layer.Name}']: ledger parameter '{lk}' was not emitted by parser.");
                }
            }
        }

        VerifyDetachedNeurons(network, root, errors);
        return checkedParams;
    }

    private static void VerifyDetachedNeurons(
        NetworkDefinition network, JsonElement root, List<string> errors)
    {
        if (!root.TryGetProperty("detached_neuron_layers", out var detEl) || detEl.ValueKind != JsonValueKind.Array)
            return;

        var ledgerDet = detEl.EnumerateArray().ToArray();
        if (ledgerDet.Length != network.DetachedNeuronLayers.Count)
        {
            errors.Add(
                $"spec-link.detached_neuron_layers: parser emitted {network.DetachedNeuronLayers.Count} detached neuron layers, ledger has {ledgerDet.Length}.");
            return;
        }

        for (int i = 0; i < ledgerDet.Length; i++)
        {
            var actual = network.DetachedNeuronLayers[i];
            var expected = ledgerDet[i];
            var expName = expected.GetProperty("name").GetString()!;
            var expSrc = expected.GetProperty("src_layer").GetString()!;
            var expNeuron = expected.GetProperty("neuron").GetString()!;
            int expGpu = expected.GetProperty("gpu").GetInt32();

            if (actual.Name != expName || actual.SourceLayer != expSrc ||
                actual.Parameters["neuron"].AsString() != expNeuron ||
                actual.Parameters["gpu"].AsInt() != expGpu)
            {
                errors.Add(
                    $"spec-link.detached_neuron_layers[{i}]: mismatch on '{actual.Name}' vs ledger '{expName}'.");
            }
        }
    }

    private static void VerifyValueMatchesJson(
        string layerName, string paramName, ConfigValue actual, JsonElement expected, List<string> errors)
    {
        bool ok = actual.Kind switch
        {
            ConfigValueKind.Int => expected.ValueKind == JsonValueKind.Number &&
                                   expected.TryGetInt32(out int iv) && iv == actual.AsInt(),
            ConfigValueKind.Float => expected.ValueKind == JsonValueKind.Number &&
                                     Math.Abs(expected.GetDouble() - actual.AsFloat()) <= 1e-15,
            ConfigValueKind.Bool => (expected.ValueKind == JsonValueKind.True && actual.AsBool()) ||
                                    (expected.ValueKind == JsonValueKind.False && !actual.AsBool()),
            ConfigValueKind.String => expected.ValueKind == JsonValueKind.String &&
                                      expected.GetString() == actual.AsString(),
            ConfigValueKind.IntList => expected.ValueKind == JsonValueKind.Array &&
                                       expected.EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(actual.AsIntList()),
            ConfigValueKind.FloatList => expected.ValueKind == JsonValueKind.Array &&
                                         FloatListsEqual(expected, actual.AsFloatList()),
            ConfigValueKind.BoolList => expected.ValueKind == JsonValueKind.Array &&
                                        expected.EnumerateArray().Select(x => x.GetBoolean()).SequenceEqual(actual.AsBoolList()),
            ConfigValueKind.StringList => expected.ValueKind == JsonValueKind.Array &&
                                          expected.EnumerateArray().Select(x => x.GetString() ?? "").SequenceEqual(actual.AsStringList(), StringComparer.Ordinal),
            _ => false,
        };

        if (!ok)
        {
            errors.Add(
                $"spec-link['{layerName}'].parameters.{paramName}: parser value {actual} != ledger {expected.GetRawText()}.");
        }
    }

    private static bool FloatListsEqual(JsonElement arr, IReadOnlyList<double> actual)
    {
        var exp = arr.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        if (exp.Length != actual.Count)
            return false;
        for (int i = 0; i < exp.Length; i++)
        {
            if (Math.Abs(exp[i] - actual[i]) > 1e-15)
                return false;
        }
        return true;
    }
}
