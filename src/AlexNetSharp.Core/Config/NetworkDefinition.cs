namespace AlexNetSharp.Core.Config;

public enum LayerKind
{
    Data,
    Conv,
    CmrNorm,
    RNorm,
    Pool,
    Fc,
    Hs,
    Softmax,
    CostLogreg,
    Neuron,
}

/// <summary>
/// Parsed, defaulted, and params-merged definition of a single network layer
/// (<c>layer.py:278-320</c>), prior to shape inference (ANX-3).
/// </summary>
public sealed class LayerDefinition
{
    public string Name { get; }
    public string Type { get; }
    public LayerKind Kind { get; }

    /// <summary>Inputs as originally declared in the layer config before neuron detachment.</summary>
    public IReadOnlyList<string> DeclaredInputs { get; }

    /// <summary>Effective inputs in order after <c>detach_neuron_layers</c> (<c>layer.py:642-645</c>).</summary>
    public IReadOnlyList<string> Inputs { get; }

    /// <summary>For a detached neuron layer (<c>&lt;src&gt;_neuron</c>), the name of its source layer.</summary>
    public string? SourceLayer { get; }

    /// <summary>If this layer declared <c>neuron=...</c>, the synthesized detached neuron layer name.</summary>
    public string? DetachedNeuronLayer { get; }

    public bool IsDetachedNeuron => SourceLayer is not null;
    public bool RequiresParams { get; }

    public LayerParameters TypedParams { get; }

    /// <summary>
    /// Canonical parameter map (sorted by Ordinal key) containing every parameter after defaults
    /// and params-file merging.
    /// </summary>
    public IReadOnlyDictionary<string, ConfigValue> Parameters { get; }

    public LayerDefinition(
        string name,
        string type,
        LayerKind kind,
        IReadOnlyList<string> declaredInputs,
        IReadOnlyList<string> inputs,
        string? sourceLayer,
        string? detachedNeuronLayer,
        bool requiresParams,
        LayerParameters typedParams,
        IReadOnlyDictionary<string, ConfigValue> parameters)
    {
        Name = name;
        Type = type;
        Kind = kind;
        DeclaredInputs = declaredInputs;
        Inputs = inputs;
        SourceLayer = sourceLayer;
        DetachedNeuronLayer = detachedNeuronLayer;
        RequiresParams = requiresParams;
        TypedParams = typedParams;
        Parameters = parameters;
    }
}

/// <summary>
/// Complete parsed network model from a <c>layers-*.cfg</c> + <c>layer-params-*.cfg</c> pair.
/// </summary>
public sealed class NetworkDefinition
{
    private readonly Dictionary<string, LayerDefinition> _byName;

    /// <summary>All layers in parser order (declared sections in file order, followed by detached neuron layers).</summary>
    public IReadOnlyList<LayerDefinition> Layers { get; }

    public IReadOnlyList<LayerDefinition> DeclaredLayers { get; }
    public IReadOnlyList<LayerDefinition> DetachedNeuronLayers { get; }
    public IReadOnlyDictionary<string, LayerDefinition> LayersByName => _byName;

    public LayerDefinition this[string name] => _byName[name];

    public NetworkDefinition(IReadOnlyList<LayerDefinition> layers)
    {
        Layers = layers.ToArray();
        DeclaredLayers = Layers.Where(l => !l.IsDetachedNeuron).ToArray();
        DetachedNeuronLayers = Layers.Where(l => l.IsDetachedNeuron).ToArray();
        _byName = new Dictionary<string, LayerDefinition>(Layers.Count, StringComparer.Ordinal);
        foreach (var layer in Layers)
            _byName[layer.Name] = layer;
    }
}
