namespace AlexNetSharp.Core.Config;

/// <summary>
/// Base class for strongly-typed per-layer parameters after defaults and params-file merging.
/// </summary>
public abstract record LayerParameters(
    int Gpu,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB);

public sealed record DataLayerParams(
    int DataIdx,
    int Gpu,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);

/// <summary>
/// Typed parameters for <c>type=conv</c> (<c>layer.py:711-871, 974-1076</c>).
/// <see cref="ConfigPadding"/> holds non-negative config values (<c>layer.py:981</c>);
/// <see cref="PaddingStart"/> holds negated <c>-padding</c> stored by <c>layer.py:1041</c>.
/// </summary>
public sealed record ConvLayerParams(
    IReadOnlyList<int> Channels,
    IReadOnlyList<int> Filters,
    int OutputFilters,
    IReadOnlyList<int> ConfigPadding,
    IReadOnlyList<int> PaddingStart,
    IReadOnlyList<int> Stride,
    IReadOnlyList<int> FilterSize,
    IReadOnlyList<int> Groups,
    IReadOnlyList<bool> RandSparse,
    IReadOnlyList<int>? ExplicitFilterChannels,
    IReadOnlyList<double> InitW,
    double InitB,
    string InitWFunc,
    string InitBFunc,
    string InitCFunc,
    IReadOnlyList<string> WeightSource,
    IReadOnlyList<string> WeightSourceLayers,
    IReadOnlyList<int> WeightSourceMatrixIndices,
    int PartialSum,
    bool SharedBiases,
    string Neuron,
    int Gpu,
    IReadOnlyList<double> EpsW,
    double EpsB,
    IReadOnlyList<double> MomW,
    double MomB,
    double SuperEps,
    double SuperMom,
    IReadOnlyList<double> Wc,
    IReadOnlyList<double> Wball,
    IReadOnlyList<double> WcNormMin,
    IReadOnlyList<double> WcNormMax,
    ScheduleSpec SchedW,
    ScheduleSpec SchedB,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB,
    bool GradConsumer) : LayerParameters(Gpu, QuantF, QuantB);

public sealed record FcLayerParams(
    int Outputs,
    IReadOnlyList<double> InitW,
    double InitB,
    string InitWFunc,
    string InitBFunc,
    IReadOnlyList<string> WeightSource,
    IReadOnlyList<string> WeightSourceLayers,
    IReadOnlyList<int> WeightSourceMatrixIndices,
    string Neuron,
    int Gpu,
    IReadOnlyList<double> EpsW,
    double EpsB,
    IReadOnlyList<double> MomW,
    double MomB,
    double SuperEps,
    double SuperMom,
    IReadOnlyList<double> Wc,
    IReadOnlyList<double> Wball,
    ScheduleSpec SchedW,
    ScheduleSpec SchedB,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB,
    bool GradConsumer) : LayerParameters(Gpu, QuantF, QuantB);

public sealed record PoolLayerParams(
    string Pool,
    int Channels,
    int SizeX,
    int Start,
    int Stride,
    int OutputsX,
    bool? DoMax,
    string Neuron,
    int Gpu,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);

/// <summary>
/// Typed parameters for <c>type=cmrnorm</c> (<c>layer.py:1170-1216</c>).
/// <see cref="RawScale"/> is the config value; <see cref="Scale"/> is divided by <see cref="Size"/> (<c>layer.py:1183</c>).
/// </summary>
public sealed record CmrNormLayerParams(
    int Channels,
    int Size,
    bool Blocked,
    string Neuron,
    int Gpu,
    double RawScale,
    double Scale,
    double Pow,
    double MinDiv,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);

/// <summary>
/// Typed parameters for <c>type=rnorm</c> (<c>layer.py:1170-1216</c>).
/// <see cref="RawScale"/> is the config value; <see cref="Scale"/> is divided by <c>Size * Size</c> (<c>layer.py:1183</c>).
/// </summary>
public sealed record RNormLayerParams(
    int Channels,
    int Size,
    bool Blocked,
    string Neuron,
    int Gpu,
    double RawScale,
    double Scale,
    double Pow,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);

public sealed record DropoutLayerParams(
    double Keep,
    bool Enable,
    string Neuron,
    int Gpu,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);

public sealed record SoftmaxLayerParams(
    string Neuron,
    int Gpu,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);

public sealed record LogregCostLayerParams(
    double Coeff,
    int TopK,
    int Gpu,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);

public sealed record NeuronLayerParams(
    string NeuronType,
    bool UsesActs,
    bool UsesInputs,
    int Gpu,
    ScheduleSpec QuantF,
    ScheduleSpec QuantB) : LayerParameters(Gpu, QuantF, QuantB);
