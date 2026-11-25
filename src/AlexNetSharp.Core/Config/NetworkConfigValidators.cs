using System.Globalization;
using System.Text.RegularExpressions;

namespace AlexNetSharp.Core.Config;

internal static class NetworkConfigValidators
{
    private static readonly IReadOnlyDictionary<string, double> EmptyParams =
        new Dictionary<string, double>(StringComparer.Ordinal);

    public static readonly HashSet<string> SupportedLayerTypes = new(StringComparer.Ordinal)
    {
        "data", "conv", "cmrnorm", "rnorm", "pool", "fc", "hs", "softmax", "cost.logreg", "neuron",
    };

    public static readonly HashSet<string> UnsupportedKnownLayerTypes = new(StringComparer.Ordinal)
    {
        "treefc", "local", "eltsum", "eltmax", "cnorm", "nailbed", "blur", "href", "resize",
        "rgb2yuv", "rgb2lab", "rscale", "concat", "cost.msm", "cost.crossent", "cost.crossent2",
        "cost.sum2", "cost.gsum2", "cost.tica", "cost.rflickr",
    };

    private static readonly HashSet<string> UnsupportedZeroParamNeurons = new(StringComparer.Ordinal)
    {
        "ident", "logistic", "abs", "nrelu", "softrelu", "square", "sqrt",
    };

    private static readonly Dictionary<string, int> UnsupportedParamNeurons = new(StringComparer.Ordinal)
    {
        ["abstanh"] = 2,
        ["tanh"] = 2,
        ["brelu"] = 1,
        ["linear"] = 2,
        ["drelu"] = 1,
    };

    private static readonly Regex WeightSourcePattern =
        new(@"^\s*([^\s\[]+)(?:\[(\d+)\])?\s*$", RegexOptions.Compiled);

    public static void ValidateNeuron(string layerName, string neuronStr)
    {
        if (neuronStr == "relu")
            return;

        if (UnsupportedZeroParamNeurons.Contains(neuronStr) || MatchesUnsupportedParamNeuron(neuronStr))
        {
            throw new LayerParsingException(
                $"Layer '{layerName}': neuron type '{neuronStr}' is unsupported in this port", layerName);
        }

        throw new LayerParsingException(
            $"Layer '{layerName}': unable to parse neuron type '{neuronStr}'. Valid neuron types: relu. Where neurons have parameters, they must be floats.",
            layerName);
    }

    private static bool MatchesUnsupportedParamNeuron(string s)
    {
        int openIdx = s.IndexOf('[');
        if (openIdx <= 0)
            return false;
        var baseName = s[..openIdx].TrimEnd();
        if (!UnsupportedParamNeurons.TryGetValue(baseName, out int expectedCount))
            return false;
        var rest = s[(openIdx + 1)..].TrimEnd();
        if (!rest.EndsWith(']'))
            return false;
        var inner = rest[..^1];
        var parts = inner.Split(',');
        if (parts.Length != expectedCount)
            return false;
        foreach (var part in parts)
        {
            if (!double.TryParse(part.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ||
                double.IsNaN(v) || double.IsInfinity(v))
            {
                return false;
            }
        }
        return true;
    }

    public static ScheduleSpec ParseQuantScheme(
        IniConfigDocument paramDoc, string layerName, string? sourceLayer, string paramName, string humanName)
    {
        var raw = LookupParamWithDetachedFallback(paramDoc, layerName, sourceLayer, paramName, "default");
        if (raw == "default")
            return new ScheduleSpec("default", "default", EmptyParams);
        if (raw == "half")
        {
            throw new LayerParsingException(
                $"Layer '{layerName}': {humanName} '{raw}' is unsupported in this port", layerName);
        }
        throw new LayerParsingException(
            $"Layer '{layerName}': unable to parse {humanName} {paramName}={raw}.", layerName);
    }

    public static ScheduleSpec ParseScheduleScheme(
        IniConfigDocument paramDoc, string layerName, string? sourceLayer, string paramName)
    {
        var raw = LookupParamWithDetachedFallback(paramDoc, layerName, sourceLayer, paramName, "default");
        if (raw == "default")
            return new ScheduleSpec("default", "default", EmptyParams);

        var trimmed = raw.Trim();
        foreach (var prefix in new[] { "linear", "exp", "dexp", "jdexp" })
        {
            if (trimmed.StartsWith(prefix + "[", StringComparison.OrdinalIgnoreCase) && trimmed.EndsWith(']'))
            {
                throw new LayerParsingException(
                    $"Layer '{layerName}': learning rate schedule '{raw}' is unsupported in this port", layerName);
            }
        }

        throw new LayerParsingException(
            $"Layer '{layerName}': unable to parse learning rate schedule {paramName}={raw}.", layerName);
    }

    private static string LookupParamWithDetachedFallback(
        IniConfigDocument paramDoc, string layerName, string? sourceLayer, string paramName, string defaultValue)
    {
        if (paramDoc.HasSection(layerName))
            return paramDoc.SafeGet(layerName, paramName, defaultValue);
        if (sourceLayer is not null && paramDoc.HasSection(sourceLayer))
            return paramDoc.SafeGet(sourceLayer, paramName, defaultValue);
        return defaultValue;
    }

    public static (string LayerName, int MatrixIdx)? TryParseWeightSource(string spec)
    {
        var m = WeightSourcePattern.Match(spec);
        if (!m.Success)
            return null;
        int idx = m.Groups[2].Success ? int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture) : 0;
        return (m.Groups[1].Value, idx);
    }

    public static void VerifyNumParams(string section, int inputCount, string paramName, int listCount)
    {
        if (listCount != inputCount)
        {
            throw new LayerParsingException(
                $"Layer '{section}': {paramName} list length does not match number of inputs", section);
        }
    }

    public static void VerifyIntRange(string section, int v, string paramName, int? min, int? max, int inputIdx = -1)
    {
        var layerName = inputIdx < 0 ? section : $"{section}[{inputIdx}]";
        if (min.HasValue && max.HasValue && (v < min.Value || v > max.Value))
        {
            throw new LayerParsingException(
                $"Layer '{layerName}': parameter '{paramName}' must be in the range {min.Value}-{max.Value}", section);
        }
        if (min.HasValue && v < min.Value)
        {
            throw new LayerParsingException(
                $"Layer '{layerName}': parameter '{paramName}' must be greater than or equal to {min.Value}", section);
        }
        if (max.HasValue && v > max.Value)
        {
            throw new LayerParsingException(
                $"Layer '{layerName}': parameter '{paramName}' must be smaller than or equal to {max.Value}", section);
        }
    }

    public static void VerifyIntListRange(string section, IReadOnlyList<int> list, string paramName, int? min, int? max)
    {
        for (int i = 0; i < list.Count; i++)
            VerifyIntRange(section, list[i], paramName, min, max, i);
    }

    public static void VerifyDivisible(
        string section, bool hasInputs, int value, int div, string valueName, string? divName = null, int inputIdx = 0)
    {
        var layerName = !hasInputs ? section : $"{section}[{inputIdx}]";
        if (value % div != 0)
        {
            var divText = divName is null ? div.ToString(CultureInfo.InvariantCulture) : $"'{divName}'";
            throw new LayerParsingException(
                $"Layer '{layerName}': parameter '{valueName}' must be divisible by {divText}", section);
        }
    }

    public static void VerifyStrIn(string section, string value, string paramName, IReadOnlyList<string> allowed)
    {
        if (!allowed.Contains(value, StringComparer.Ordinal))
        {
            var listStr = string.Join(", ", allowed.Select(s => $"'{s}'"));
            throw new LayerParsingException(
                $"Layer '{section}': parameter '{paramName}' must be one of {listStr}", section);
        }
    }
}
