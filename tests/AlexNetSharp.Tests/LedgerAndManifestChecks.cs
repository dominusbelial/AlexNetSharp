using System.Text.Json;

namespace AlexNetSharp.Tests;

internal static class LedgerAndManifestChecks
{
    private static readonly HashSet<string> WeightedLayerTypes = ["conv", "fc"];

    public static (int SectionCount, int TypeCount, int CitationCount) VerifyLedger(
        string origRoot,
        string ledgerPath,
        Dictionary<string, string[]> fileCache,
        List<SpecVerifier.CfgSection> cfgSections,
        List<string> errors)
    {
        if (!File.Exists(ledgerPath))
        {
            errors.Add($"Ledger file not found: {ledgerPath}");
            return (0, 0, 0);
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(ledgerPath));
        var root = doc.RootElement;

        int citations = SpecVerifier.VerifyAllCitations(root, origRoot, fileCache, "ledger", errors);
        if (citations == 0)
            errors.Add("ledger: expected at least one citation object.");

        if (!root.TryGetProperty("layer_types", out var layerTypesEl) || layerTypesEl.ValueKind != JsonValueKind.Object)
        {
            errors.Add("ledger: missing object 'layer_types'.");
            return (0, 0, citations);
        }

        var usedTypes = cfgSections.Select(s => s.Type).Where(t => t.Length > 0).ToHashSet(StringComparer.Ordinal);
        if (cfgSections.Any(s => !string.IsNullOrEmpty(s.Neuron)))
            usedTypes.Add("neuron");

        foreach (var t in usedTypes.OrderBy(x => x, StringComparer.Ordinal))
        {
            if (!layerTypesEl.TryGetProperty(t, out var tEntry) || tEntry.ValueKind != JsonValueKind.Object)
            {
                errors.Add($"ledger.layer_types: missing entry for used layer type '{t}'.");
                continue;
            }
            RequireNonEmptyString(tEntry, "forward_semantics", $"ledger.layer_types['{t}']", errors);
            RequireNonEmptyString(tEntry, "backward_semantics", $"ledger.layer_types['{t}']", errors);
            RequireNonEmptyArray(tEntry, "citations", $"ledger.layer_types['{t}']", errors);
        }

        var usedNeurons = cfgSections.Select(s => s.Neuron).Where(n => n.Length > 0).ToHashSet(StringComparer.Ordinal);
        if (!root.TryGetProperty("neuron_types", out var neuronTypesEl) || neuronTypesEl.ValueKind != JsonValueKind.Object)
        {
            errors.Add("ledger: missing object 'neuron_types'.");
        }
        else
        {
            foreach (var n in usedNeurons.OrderBy(x => x, StringComparer.Ordinal))
            {
                var baseNeuron = n.Contains('[') ? n[..n.IndexOf('[')].Trim() : n;
                if (!neuronTypesEl.TryGetProperty(baseNeuron, out _))
                    errors.Add($"ledger.neuron_types: missing entry for used neuron type '{baseNeuron}'.");
            }
        }

        if (!root.TryGetProperty("sections", out var sectionsEl) || sectionsEl.ValueKind != JsonValueKind.Array)
        {
            errors.Add("ledger: missing array 'sections'.");
            return (0, usedTypes.Count, citations);
        }

        var byName = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var sec in sectionsEl.EnumerateArray())
        {
            var name = sec.TryGetProperty("name", out var nEl) && nEl.ValueKind == JsonValueKind.String ? nEl.GetString()! : "";
            if (name.Length == 0)
                errors.Add("ledger.sections: entry missing string 'name'.");
            else if (!byName.TryAdd(name, sec))
                errors.Add($"ledger.sections: duplicate section entry '{name}'.");
        }

        foreach (var cfgSec in cfgSections)
        {
            if (!byName.TryGetValue(cfgSec.Name, out var entry))
            {
                errors.Add($"ledger.sections: missing entry for layers-145.cfg section '[{cfgSec.Name}]'.");
                continue;
            }
            var entryType = entry.TryGetProperty("type", out var tp) && tp.ValueKind == JsonValueKind.String ? tp.GetString()! : "";
            if (entryType != cfgSec.Type)
                errors.Add($"ledger.sections['{cfgSec.Name}']: type '{entryType}' != cfg type '{cfgSec.Type}'.");

            if (!entry.TryGetProperty("inputs", out var inpArr) || inpArr.ValueKind != JsonValueKind.Array)
            {
                errors.Add($"ledger.sections['{cfgSec.Name}']: missing 'inputs' array.");
            }
            else
            {
                var ledgerInputs = inpArr.EnumerateArray().Select(x => x.GetString() ?? "").ToList();
                if (!ledgerInputs.SequenceEqual(cfgSec.Inputs, StringComparer.Ordinal))
                    errors.Add($"ledger.sections['{cfgSec.Name}']: inputs [{string.Join(",", ledgerInputs)}] != cfg [{string.Join(",", cfgSec.Inputs)}].");
            }

            if (!entry.TryGetProperty("parameters", out var pObj) || pObj.ValueKind != JsonValueKind.Object || !pObj.EnumerateObject().Any())
            {
                errors.Add($"ledger.sections['{cfgSec.Name}']: missing or empty 'parameters' object.");
            }
            else
            {
                foreach (var param in pObj.EnumerateObject())
                {
                    if (param.Value.ValueKind != JsonValueKind.Object ||
                        !param.Value.TryGetProperty("value", out _) ||
                        !param.Value.TryGetProperty("default", out _))
                    {
                        errors.Add($"ledger.sections['{cfgSec.Name}'].parameters.{param.Name}: must contain 'value' and 'default'.");
                    }
                }
            }
            RequireNonEmptyString(entry, "forward_semantics", $"ledger.sections['{cfgSec.Name}']", errors);
            RequireNonEmptyString(entry, "backward_semantics", $"ledger.sections['{cfgSec.Name}']", errors);
            RequireNonEmptyArray(entry, "citations", $"ledger.sections['{cfgSec.Name}']", errors);
        }

        return (byName.Count, usedTypes.Count, citations);
    }

    public static (int WeightedCount, int CitationCount) VerifyManifest(
        string origRoot,
        string manifestPath,
        Dictionary<string, string[]> fileCache,
        List<SpecVerifier.CfgSection> cfgSections,
        List<string> errors)
    {
        if (!File.Exists(manifestPath))
        {
            errors.Add($"Experiment manifest file not found: {manifestPath}");
            return (0, 0);
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var root = doc.RootElement;
        int citations = SpecVerifier.VerifyAllCitations(root, origRoot, fileCache, "manifest", errors);
        if (citations == 0)
            errors.Add("manifest: expected at least one citation object.");

        var weightedSections = cfgSections.Where(s => WeightedLayerTypes.Contains(s.Type)).ToList();
        VerifyRateMap(root, "recorded_layer_rates", weightedSections, requirePositiveStartEpsW: false, errors);

        if (!root.TryGetProperty("proposed_fresh_run", out var freshEl) || freshEl.ValueKind != JsonValueKind.Object)
            errors.Add("manifest: missing object 'proposed_fresh_run'.");
        else
            VerifyRateMap(freshEl, "layer_rates", weightedSections, requirePositiveStartEpsW: true, errors);

        return (weightedSections.Count, citations);
    }

    private static void VerifyRateMap(
        JsonElement parent,
        string propName,
        List<SpecVerifier.CfgSection> weightedSections,
        bool requirePositiveStartEpsW,
        List<string> errors)
    {
        if (!parent.TryGetProperty(propName, out var ratesObj) || ratesObj.ValueKind != JsonValueKind.Object)
        {
            errors.Add($"manifest: missing per-layer rates object '{propName}'.");
            return;
        }
        foreach (var sec in weightedSections)
        {
            if (!ratesObj.TryGetProperty(sec.Name, out var r) || r.ValueKind != JsonValueKind.Object)
            {
                errors.Add($"manifest.{propName}: missing rates entry for weighted layer '{sec.Name}'.");
                continue;
            }
            int expectedCount = Math.Max(1, sec.Inputs.Count);
            CheckFloatList(r, "epsW", expectedCount, requirePositiveStartEpsW, $"manifest.{propName}['{sec.Name}']", errors);
            CheckFloatList(r, "momW", expectedCount, false, $"manifest.{propName}['{sec.Name}']", errors);
            CheckFloatList(r, "wc", expectedCount, false, $"manifest.{propName}['{sec.Name}']", errors);
            CheckNumber(r, "epsB", $"manifest.{propName}['{sec.Name}']", errors);
            CheckNumber(r, "momB", $"manifest.{propName}['{sec.Name}']", errors);
        }
    }

    private static void CheckFloatList(JsonElement obj, string prop, int expectedLen, bool requirePositive, string where, List<string> errors)
    {
        if (!obj.TryGetProperty(prop, out var arr) || arr.ValueKind != JsonValueKind.Array)
        {
            errors.Add($"{where}: missing numeric array '{prop}'.");
            return;
        }
        var vals = arr.EnumerateArray().ToList();
        if (vals.Count != expectedLen)
            errors.Add($"{where}.{prop}: length {vals.Count} != expected {expectedLen}.");
        foreach (var v in vals)
        {
            if (v.ValueKind != JsonValueKind.Number)
                errors.Add($"{where}.{prop}: non-numeric element.");
            else if (requirePositive && v.GetDouble() <= 0.0)
                errors.Add($"{where}.{prop}: fresh-run starting rate must be > 0 (got {v.GetDouble()}).");
        }
    }

    private static void CheckNumber(JsonElement obj, string prop, string where, List<string> errors)
    {
        if (!obj.TryGetProperty(prop, out var v) || v.ValueKind != JsonValueKind.Number)
            errors.Add($"{where}: missing numeric property '{prop}'.");
    }

    private static void RequireNonEmptyString(JsonElement obj, string prop, string where, List<string> errors)
    {
        if (!obj.TryGetProperty(prop, out var v) || v.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(v.GetString()))
            errors.Add($"{where}: missing or empty string '{prop}'.");
    }

    private static void RequireNonEmptyArray(JsonElement obj, string prop, string where, List<string> errors)
    {
        if (!obj.TryGetProperty(prop, out var v) || v.ValueKind != JsonValueKind.Array || v.GetArrayLength() == 0)
            errors.Add($"{where}: missing or empty array '{prop}'.");
    }
}
