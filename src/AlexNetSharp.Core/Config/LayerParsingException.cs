namespace AlexNetSharp.Core.Config;

/// <summary>
/// Exception thrown when parsing network layer or parameter configuration files fails.
/// Mirrors <c>LayerParsingError</c> in <c>layer.py:13</c>.
/// </summary>
public sealed class LayerParsingException : Exception
{
    public string? SectionName { get; }

    public LayerParsingException(string message, string? sectionName = null)
        : base(message)
    {
        SectionName = sectionName;
    }

    public LayerParsingException(string message, Exception innerException, string? sectionName = null)
        : base(message, innerException)
    {
        SectionName = sectionName;
    }
}
