using System.Globalization;

namespace AlexNetSharp.Core.Config;

public enum ConfigValueKind
{
    Int,
    Float,
    Bool,
    String,
    IntList,
    FloatList,
    BoolList,
    StringList,
}

/// <summary>
/// Parsed parameter-scheme specification for <c>schedW</c>, <c>schedB</c>, <c>quantF</c>, <c>quantB</c>
/// (<c>ParamParser</c> in <c>layer.py:68-93, 157-178, 739-740</c>).
/// </summary>
public sealed record ScheduleSpec(
    string Raw,
    string Type,
    IReadOnlyDictionary<string, double> Parameters);

/// <summary>
/// Discriminated union representing a single defaulted or merged layer parameter value.
/// </summary>
public sealed class ConfigValue : IEquatable<ConfigValue>
{
    private readonly object _value;

    public ConfigValueKind Kind { get; }

    private ConfigValue(ConfigValueKind kind, object value)
    {
        Kind = kind;
        _value = value;
    }

    public static ConfigValue FromInt(int v) => new(ConfigValueKind.Int, v);
    public static ConfigValue FromFloat(double v) => new(ConfigValueKind.Float, v);
    public static ConfigValue FromBool(bool v) => new(ConfigValueKind.Bool, v);
    public static ConfigValue FromString(string v) => new(ConfigValueKind.String, v);
    public static ConfigValue FromIntList(IReadOnlyList<int> v) => new(ConfigValueKind.IntList, v.ToArray());
    public static ConfigValue FromFloatList(IReadOnlyList<double> v) => new(ConfigValueKind.FloatList, v.ToArray());
    public static ConfigValue FromBoolList(IReadOnlyList<bool> v) => new(ConfigValueKind.BoolList, v.ToArray());
    public static ConfigValue FromStringList(IReadOnlyList<string> v) => new(ConfigValueKind.StringList, v.ToArray());

    public int AsInt() => Kind == ConfigValueKind.Int ? (int)_value : throw BadKind(ConfigValueKind.Int);
    public double AsFloat() => Kind == ConfigValueKind.Float ? (double)_value : throw BadKind(ConfigValueKind.Float);
    public bool AsBool() => Kind == ConfigValueKind.Bool ? (bool)_value : throw BadKind(ConfigValueKind.Bool);
    public string AsString() => Kind == ConfigValueKind.String ? (string)_value : throw BadKind(ConfigValueKind.String);
    public IReadOnlyList<int> AsIntList() => Kind == ConfigValueKind.IntList ? (int[])_value : throw BadKind(ConfigValueKind.IntList);
    public IReadOnlyList<double> AsFloatList() => Kind == ConfigValueKind.FloatList ? (double[])_value : throw BadKind(ConfigValueKind.FloatList);
    public IReadOnlyList<bool> AsBoolList() => Kind == ConfigValueKind.BoolList ? (bool[])_value : throw BadKind(ConfigValueKind.BoolList);
    public IReadOnlyList<string> AsStringList() => Kind == ConfigValueKind.StringList ? (string[])_value : throw BadKind(ConfigValueKind.StringList);

    private InvalidOperationException BadKind(ConfigValueKind expected) =>
        new($"Expected {expected}, but ConfigValue is {Kind}.");

    public bool Equals(ConfigValue? other)
    {
        if (other is null || Kind != other.Kind)
            return false;
        return Kind switch
        {
            ConfigValueKind.Int => AsInt() == other.AsInt(),
            ConfigValueKind.Float => AsFloat().Equals(other.AsFloat()),
            ConfigValueKind.Bool => AsBool() == other.AsBool(),
            ConfigValueKind.String => string.Equals(AsString(), other.AsString(), StringComparison.Ordinal),
            ConfigValueKind.IntList => AsIntList().SequenceEqual(other.AsIntList()),
            ConfigValueKind.FloatList => AsFloatList().SequenceEqual(other.AsFloatList()),
            ConfigValueKind.BoolList => AsBoolList().SequenceEqual(other.AsBoolList()),
            ConfigValueKind.StringList => AsStringList().SequenceEqual(other.AsStringList(), StringComparer.Ordinal),
            _ => false,
        };
    }

    public override bool Equals(object? obj) => Equals(obj as ConfigValue);

    public override int GetHashCode() => HashCode.Combine(Kind);

    public override string ToString() => Kind switch
    {
        ConfigValueKind.Int => AsInt().ToString(CultureInfo.InvariantCulture),
        ConfigValueKind.Float => AsFloat().ToString("G17", CultureInfo.InvariantCulture),
        ConfigValueKind.Bool => AsBool() ? "true" : "false",
        ConfigValueKind.String => $"\"{AsString()}\"",
        ConfigValueKind.IntList => $"[{string.Join(", ", AsIntList().Select(x => x.ToString(CultureInfo.InvariantCulture)))}]",
        ConfigValueKind.FloatList => $"[{string.Join(", ", AsFloatList().Select(x => x.ToString("G17", CultureInfo.InvariantCulture)))}]",
        ConfigValueKind.BoolList => $"[{string.Join(", ", AsBoolList().Select(x => x ? "true" : "false"))}]",
        ConfigValueKind.StringList => $"[{string.Join(", ", AsStringList().Select(x => $"\"{x}\""))}]",
        _ => "",
    };
}
