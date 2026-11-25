using System.Globalization;
using System.Text;

namespace AlexNetSharp.Core.Config;

/// <summary>
/// Minimal INI reader reproducing Python 2 <c>SafeConfigParser(dict_type=OrderedDict)</c>
/// and <c>MyConfigParser</c> (<c>layer.py:95-133, 285-286, 307-308</c>).
/// </summary>
public sealed class IniConfigDocument
{
    public const string DefaultSectionName = "DEFAULT";
    private const int MaxInterpolationDepth = 10;

    private readonly List<string> _sectionOrder = [];
    private readonly Dictionary<string, Dictionary<string, string>> _sections = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _defaults = new(StringComparer.Ordinal);

    public string SourceName { get; }
    public IReadOnlyList<string> Sections => _sectionOrder;

    private IniConfigDocument(string sourceName)
    {
        SourceName = sourceName;
    }

    public static IniConfigDocument Load(string filePath)
    {
        var content = File.ReadAllText(filePath);
        return Parse(content, filePath);
    }

    public static IniConfigDocument Parse(string content, string sourceName = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(content);
        var doc = new IniConfigDocument(sourceName);
        doc.ReadLines(content);
        return doc;
    }

    public bool HasSection(string section) => _sections.ContainsKey(section);

    public bool HasOption(string section, string option)
    {
        var key = NormalizeOption(option);
        if (section == DefaultSectionName)
            return _defaults.ContainsKey(key);
        return _sections.TryGetValue(section, out var dict) && (dict.ContainsKey(key) || _defaults.ContainsKey(key));
    }

    private static string NormalizeOption(string option) => option.ToLowerInvariant();

    private void ReadLines(string content)
    {
        Dictionary<string, string>? currentSection = null;
        string? currentSectionName = null;
        string? currentOption = null;

        using var reader = new StringReader(content);
        int lineNo = 0;
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            lineNo++;
            var trimmedStart = line.TrimStart();
            if (trimmedStart.Length == 0 || trimmedStart[0] == '#' || trimmedStart[0] == ';')
                continue;

            if (char.IsWhiteSpace(line[0]))
            {
                if (currentSection is null || currentOption is null)
                {
                    throw new LayerParsingException(
                        $"INI syntax error in '{SourceName}' line {lineNo}: unexpected continuation line.");
                }
                var contVal = StripInlineSemicolonComment(trimmedStart).TrimEnd();
                if (contVal.Length > 0)
                    currentSection[currentOption] = currentSection[currentOption] + "\n" + contVal;
                continue;
            }

            if (line[0] == '[')
            {
                int closeIdx = line.IndexOf(']');
                if (closeIdx <= 1)
                {
                    throw new LayerParsingException(
                        $"INI syntax error in '{SourceName}' line {lineNo}: malformed section header '{line}'.");
                }
                var trailing = line[(closeIdx + 1)..].Trim();
                if (trailing.Length > 0 && trailing[0] != '#' && trailing[0] != ';')
                {
                    throw new LayerParsingException(
                        $"INI syntax error in '{SourceName}' line {lineNo}: unexpected text after section header '{line}'.");
                }

                var header = line[1..closeIdx];
                currentSectionName = header;
                currentOption = null;
                if (header == DefaultSectionName)
                {
                    currentSection = _defaults;
                }
                else
                {
                    if (!_sections.TryGetValue(header, out currentSection))
                    {
                        currentSection = new Dictionary<string, string>(StringComparer.Ordinal);
                        _sections[header] = currentSection;
                        _sectionOrder.Add(header);
                    }
                }
                continue;
            }

            if (currentSection is null)
            {
                throw new LayerParsingException(
                    $"INI syntax error in '{SourceName}' line {lineNo}: option '{line}' appears before any [section] header.");
            }

            int delimIdx = line.IndexOfAny(['=', ':']);
            if (delimIdx <= 0)
            {
                throw new LayerParsingException(
                    $"INI syntax error in '{SourceName}' line {lineNo}: expected 'key=value' or 'key: value', got '{line}'.",
                    currentSectionName);
            }

            var rawKey = line[..delimIdx].TrimEnd();
            if (rawKey.Length == 0)
            {
                throw new LayerParsingException(
                    $"INI syntax error in '{SourceName}' line {lineNo}: empty option name.",
                    currentSectionName);
            }

            var normKey = NormalizeOption(rawKey);
            var afterDelim = StripInlineSemicolonComment(line[(delimIdx + 1)..]).Trim();
            currentSection[normKey] = afterDelim;
            currentOption = normKey;
        }
    }

    private static string StripInlineSemicolonComment(string raw)
    {
        for (int i = 1; i < raw.Length; i++)
        {
            if (raw[i] == ';' && char.IsWhiteSpace(raw[i - 1]))
                return raw[..i];
        }
        return raw;
    }

    private bool TryGetRawOption(string section, string normalizedOption, out string rawValue)
    {
        if (section != DefaultSectionName &&
            _sections.TryGetValue(section, out var secDict) &&
            secDict.TryGetValue(normalizedOption, out rawValue!))
        {
            return true;
        }
        return _defaults.TryGetValue(normalizedOption, out rawValue!);
    }

    private string InterpolateSome(string section, string option, string rest, int depth)
    {
        if (depth > MaxInterpolationDepth)
        {
            throw new LayerParsingException(
                $"Layer '{section}': parameter '{option}' interpolation depth exceeded maximum ({MaxInterpolationDepth})",
                section);
        }

        var sb = new StringBuilder();
        int idx = 0;
        while (idx < rest.Length)
        {
            int pct = rest.IndexOf('%', idx);
            if (pct < 0)
            {
                sb.Append(rest, idx, rest.Length - idx);
                break;
            }
            sb.Append(rest, idx, pct - idx);
            if (pct + 1 >= rest.Length)
            {
                throw new LayerParsingException(
                    $"Layer '{section}': bad interpolation syntax in '{option}' ('%' must be followed by '%' or '(')",
                    section);
            }

            char next = rest[pct + 1];
            if (next == '%')
            {
                sb.Append('%');
                idx = pct + 2;
                continue;
            }

            if (next != '(')
            {
                throw new LayerParsingException(
                    $"Layer '{section}': bad interpolation syntax in '{option}' near '{rest[pct..]}'",
                    section);
            }

            int closeParen = rest.IndexOf(')', pct + 2);
            if (closeParen < 0 || closeParen + 1 >= rest.Length || rest[closeParen + 1] != 's')
            {
                throw new LayerParsingException(
                    $"Layer '{section}': incomplete or unsupported interpolation syntax in '{option}' near '{rest[pct..]}'",
                    section);
            }

            var refKey = NormalizeOption(rest[(pct + 2)..closeParen]);
            if (refKey.Length == 0 || !TryGetRawOption(section, refKey, out var refRaw))
            {
                throw new LayerParsingException(
                    $"Layer '{section}': interpolation option '{refKey}' not found when resolving '{option}'",
                    section);
            }

            sb.Append(InterpolateSome(section, option, refRaw, depth + 1));
            idx = closeParen + 2;
        }

        return sb.ToString();
    }

    public string SafeGet(string section, string option, string? defaultValue = null)
    {
        var norm = NormalizeOption(option);
        if (!_sections.ContainsKey(section) && section != DefaultSectionName)
        {
            if (defaultValue is not null)
                return defaultValue;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        if (!TryGetRawOption(section, norm, out var raw))
        {
            if (defaultValue is not null)
                return defaultValue;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        return raw.Contains('%') ? InterpolateSome(section, option, raw, depth: 1) : raw;
    }

    public int SafeGetInt(string section, string option, int? defaultValue = null)
    {
        if (!HasOption(section, option))
        {
            if (defaultValue.HasValue)
                return defaultValue.Value;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        var raw = SafeGet(section, option);
        if (int.TryParse(raw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int result))
            return result;

        throw new LayerParsingException($"Layer '{section}': parameter '{option}' must be int", section);
    }

    public double SafeGetFloat(string section, string option, double? defaultValue = null)
    {
        if (!HasOption(section, option))
        {
            if (defaultValue.HasValue)
                return defaultValue.Value;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        var raw = SafeGet(section, option);
        if (double.TryParse(raw.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double result) &&
            !double.IsNaN(result) && !double.IsInfinity(result))
        {
            return result;
        }

        throw new LayerParsingException($"Layer '{section}': parameter '{option}' must be float", section);
    }

    public bool SafeGetBool(string section, string option, bool? defaultValue = null)
    {
        if (!HasOption(section, option))
        {
            if (defaultValue.HasValue)
                return defaultValue.Value;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        var raw = SafeGet(section, option).Trim();
        if (raw.Equals("1", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("yes", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("true", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("on", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (raw.Equals("0", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("no", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("false", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("off", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        throw new LayerParsingException($"Layer '{section}': parameter '{option}' must be bool", section);
    }

    public IReadOnlyList<string> SafeGetStringList(string section, string option, IReadOnlyList<string>? defaultValue = null)
    {
        if (!HasOption(section, option))
        {
            if (defaultValue is not null)
                return defaultValue;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        var raw = SafeGet(section, option);
        var parts = raw.Split(',');
        var list = new string[parts.Length];
        for (int i = 0; i < parts.Length; i++)
            list[i] = parts[i].Trim();
        return list;
    }

    public IReadOnlyList<int> SafeGetIntList(string section, string option, IReadOnlyList<int>? defaultValue = null)
    {
        if (!HasOption(section, option))
        {
            if (defaultValue is not null)
                return defaultValue;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        var raw = SafeGet(section, option);
        var parts = raw.Split(',');
        var list = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!int.TryParse(parts[i].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out list[i]))
            {
                throw new LayerParsingException(
                    $"Layer '{section}': parameter '{option}' must be ','-delimited list of ints", section);
            }
        }
        return list;
    }

    public IReadOnlyList<double> SafeGetFloatList(string section, string option, IReadOnlyList<double>? defaultValue = null)
    {
        if (!HasOption(section, option))
        {
            if (defaultValue is not null)
                return defaultValue;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        var raw = SafeGet(section, option);
        var parts = raw.Split(',');
        var list = new double[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!double.TryParse(parts[i].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out list[i]) ||
                double.IsNaN(list[i]) || double.IsInfinity(list[i]))
            {
                throw new LayerParsingException(
                    $"Layer '{section}': parameter '{option}' must be ','-delimited list of floats", section);
            }
        }
        return list;
    }

    /// <summary>
    /// Reproduces <c>WeightLayerParser.parse</c> (<c>layer.py:832</c> vs <c>:113</c>), where
    /// <c>default=0.01</c> (a scalar float instead of a list) causes <c>0.01.split(',')</c> to fail
    /// with <c>must be ','-delimited list of floats</c> when <c>initW</c> is absent.
    /// </summary>
    public IReadOnlyList<double> SafeGetFloatListWithScalarDefaultQuirk(string section, string option)
    {
        if (!HasOption(section, option))
        {
            throw new LayerParsingException(
                $"Layer '{section}': parameter '{option}' must be ','-delimited list of floats", section);
        }
        return SafeGetFloatList(section, option);
    }

    public IReadOnlyList<bool> SafeGetBoolList(string section, string option, IReadOnlyList<bool>? defaultValue = null)
    {
        if (!HasOption(section, option))
        {
            if (defaultValue is not null)
                return defaultValue;
            throw new LayerParsingException($"Layer '{section}': required parameter '{option}' missing", section);
        }

        var raw = SafeGet(section, option);
        var parts = raw.Split(',');
        var list = new bool[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            var item = parts[i].Trim();
            list[i] = item.Equals("true", StringComparison.OrdinalIgnoreCase) || item == "1";
        }
        return list;
    }
}
