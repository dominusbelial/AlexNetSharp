namespace AlexNetSharp.Core.Config;

/// <summary>
/// Parses a <c>layers-*.cfg</c> and <c>layer-params-*.cfg</c> pair into a <see cref="NetworkDefinition"/>,
/// faithfully reproducing <c>LayerParser.parse_layers</c> (<c>layer.py:278-320</c>) without shape inference.
/// </summary>
public static class NetworkConfigParser
{
    private static readonly string[] AllowedPoolKinds = ["max", "maxabs", "avg", "rand"];

    private sealed class LayerDraft
    {
        public required string Name { get; init; }
        public required string Type { get; init; }
        public required LayerKind Kind { get; init; }
        public List<string> DeclaredInputs { get; init; } = [];
        public List<string> Inputs { get; init; } = [];
        public string? SourceLayer { get; init; }
        public string? DetachedNeuronLayer { get; set; }
        public bool RequiresParams { get; set; }
        public int Gpu { get; set; } = -1;
        public string Neuron { get; set; } = "";

        // Per-type cfg fields
        public int DataIdx { get; set; }
        public IReadOnlyList<double> InitW { get; set; } = [];
        public double InitB { get; set; }
        public string InitWFunc { get; set; } = "";
        public string InitBFunc { get; set; } = "";
        public string InitCFunc { get; set; } = "";
        public IReadOnlyList<string> WeightSource { get; set; } = [];
        public IReadOnlyList<string> WeightSourceLayers { get; set; } = [];
        public IReadOnlyList<int> WeightSourceMatrixIndices { get; set; } = [];
        public int FcOutputs { get; set; }
        public IReadOnlyList<int> ConvChannels { get; set; } = [];
        public IReadOnlyList<int> ConvFilters { get; set; } = [];
        public int ConvOutputFilters { get; set; }
        public IReadOnlyList<int> ConvConfigPadding { get; set; } = [];
        public IReadOnlyList<int> ConvPaddingStart { get; set; } = [];
        public IReadOnlyList<int> ConvStride { get; set; } = [];
        public IReadOnlyList<int> ConvFilterSize { get; set; } = [];
        public IReadOnlyList<int> ConvGroups { get; set; } = [];
        public IReadOnlyList<bool> ConvRandSparse { get; set; } = [];
        public IReadOnlyList<int>? ConvExplicitFilterChannels { get; set; }
        public int PartialSum { get; set; }
        public bool SharedBiases { get; set; } = true;
        public int SingleChannels { get; set; }
        public int PoolSizeX { get; set; }
        public int PoolStart { get; set; }
        public int PoolStride { get; set; }
        public int PoolOutputsX { get; set; }
        public string PoolKind { get; set; } = "";
        public int NormSize { get; set; }
        public bool NormBlocked { get; set; }
        public double Keep { get; set; }
    }

    public static NetworkDefinition ParseFiles(string layerCfgPath, string paramCfgPath)
    {
        if (!File.Exists(layerCfgPath))
            throw new LayerParsingException($"Layer definition file '{layerCfgPath}' does not exist");
        if (!File.Exists(paramCfgPath))
            throw new LayerParsingException($"Layer parameter file '{paramCfgPath}' does not exist");

        var layerContent = File.ReadAllText(layerCfgPath);
        var paramContent = File.ReadAllText(paramCfgPath);
        return Parse(layerContent, paramContent, layerCfgPath, paramCfgPath);
    }

    public static NetworkDefinition Parse(
        string layerCfgContent,
        string paramCfgContent,
        string layerSourceName = "<layers>",
        string paramSourceName = "<params>")
    {
        var layerDoc = IniConfigDocument.Parse(layerCfgContent, layerSourceName);
        var drafts = new List<LayerDraft>();
        var draftByName = new Dictionary<string, LayerDraft>(StringComparer.Ordinal);

        foreach (var name in layerDoc.Sections)
        {
            if (!layerDoc.HasOption(name, "type"))
                throw new LayerParsingException($"Layer '{name}': no type given", name);

            var ltype = layerDoc.SafeGet(name, "type");
            if (NetworkConfigValidators.UnsupportedKnownLayerTypes.Contains(ltype))
                throw new LayerParsingException($"Layer '{name}': layer type '{ltype}' is unsupported in this port", name);
            if (!NetworkConfigValidators.SupportedLayerTypes.Contains(ltype))
                throw new LayerParsingException($"Layer '{name}': Unknown layer type: '{ltype}'", name);

            var draft = ParseSection(name, ltype, layerDoc, draftByName);
            drafts.Add(draft);
            draftByName[name] = draft;
        }

        DetachNeuronLayers(drafts, draftByName);

        var usedInputs = new HashSet<string>(StringComparer.Ordinal);
        foreach (var d in drafts)
        {
            foreach (var inp in d.Inputs)
                usedInputs.Add(inp);
        }

        foreach (var d in drafts)
        {
            if (!d.Type.StartsWith("cost.", StringComparison.Ordinal) && !usedInputs.Contains(d.Name))
                throw new LayerParsingException($"Layer '{d.Name}' of type '{d.Type}' is unused", d.Name);
        }

        var paramDoc = IniConfigDocument.Parse(paramCfgContent, paramSourceName);
        var finalized = new List<LayerDefinition>(drafts.Count);
        foreach (var d in drafts)
        {
            if (!paramDoc.HasSection(d.Name) && d.RequiresParams)
            {
                throw new LayerParsingException(
                    $"Layer '{d.Name}' of type '{d.Type}' requires extra parameters, but none given in file '{paramSourceName}'.",
                    d.Name);
            }
            finalized.Add(FinalizeLayer(d, paramDoc));
        }

        return new NetworkDefinition(finalized);
    }

    private static void DetachNeuronLayers(List<LayerDraft> drafts, Dictionary<string, LayerDraft> draftByName)
    {
        var snapshot = drafts.ToArray();
        foreach (var src in snapshot)
        {
            if (src.Kind == LayerKind.Neuron || string.IsNullOrEmpty(src.Neuron))
                continue;

            NetworkConfigValidators.ValidateNeuron(src.Name, src.Neuron);
            var detName = GetUnusedLayerName(draftByName, $"{src.Name}_neuron");
            src.DetachedNeuronLayer = detName;

            foreach (var existing in drafts)
            {
                for (int i = 0; i < existing.Inputs.Count; i++)
                {
                    if (existing.Inputs[i] == src.Name)
                        existing.Inputs[i] = detName;
                }
            }

            var det = new LayerDraft
            {
                Name = detName,
                Type = "neuron",
                Kind = LayerKind.Neuron,
                DeclaredInputs = [src.Name],
                Inputs = [src.Name],
                SourceLayer = src.Name,
                RequiresParams = false,
                Gpu = src.Gpu,
                Neuron = src.Neuron,
            };
            drafts.Add(det);
            draftByName[detName] = det;
        }
    }

    private static string GetUnusedLayerName(Dictionary<string, LayerDraft> layers, string wish)
    {
        if (!layers.ContainsKey(wish))
            return wish;
        for (int i = 1; i < 100; i++)
        {
            var candidate = $"{wish}.{i}";
            if (!layers.ContainsKey(candidate))
                return candidate;
        }
        throw new LayerParsingException("This is @#$%&!.", wish);
    }

    private static LayerDraft ParseSection(
        string name,
        string ltype,
        IniConfigDocument doc,
        Dictionary<string, LayerDraft> prevLayers)
    {
        switch (ltype)
        {
            case "data":
                return new LayerDraft
                {
                    Name = name,
                    Type = ltype,
                    Kind = LayerKind.Data,
                    RequiresParams = false,
                    Gpu = -1,
                    DataIdx = doc.SafeGetInt(name, "dataIdx"),
                };

            case "neuron":
            {
                var d = ParseLayerWithInput(name, ltype, LayerKind.Neuron, doc, prevLayers, requiredNumInputs: 1);
                NetworkConfigValidators.ValidateNeuron(name, d.Neuron);
                return d;
            }

            case "softmax":
                return ParseLayerWithInput(name, ltype, LayerKind.Softmax, doc, prevLayers, requiredNumInputs: 1);

            case "hs":
            {
                var d = ParseLayerWithInput(name, ltype, LayerKind.Hs, doc, prevLayers, requiredNumInputs: 1);
                d.RequiresParams = true;
                d.Keep = doc.SafeGetFloat(name, "keep");
                return d;
            }

            case "fc":
            {
                var d = ParseLayerWithInput(name, ltype, LayerKind.Fc, doc, prevLayers, requiredNumInputs: -1);
                ParseWeightLayerBase(d, doc, prevLayers);
                d.FcOutputs = doc.SafeGetInt(name, "outputs");
                NetworkConfigValidators.VerifyIntRange(name, d.FcOutputs, "outputs", 1, null);
                return d;
            }

            case "conv":
            {
                var d = ParseLayerWithInput(name, ltype, LayerKind.Conv, doc, prevLayers, requiredNumInputs: -1);
                ParseConvLayer(d, doc, prevLayers);
                return d;
            }

            case "pool":
            {
                var d = ParseLayerWithInput(name, ltype, LayerKind.Pool, doc, prevLayers, requiredNumInputs: 1);
                ParsePoolLayer(d, doc);
                return d;
            }

            case "cmrnorm":
            case "rnorm":
            {
                var kind = ltype == "cmrnorm" ? LayerKind.CmrNorm : LayerKind.RNorm;
                var d = ParseLayerWithInput(name, ltype, kind, doc, prevLayers, requiredNumInputs: 1);
                ParseNormLayer(d, doc, isCrossMap: kind == LayerKind.CmrNorm);
                return d;
            }

            case "cost.logreg":
            {
                var d = ParseLayerWithInput(name, ltype, LayerKind.CostLogreg, doc, prevLayers, requiredNumInputs: 2);
                d.RequiresParams = true;
                d.Neuron = ""; // CostParser.parse deletes 'neuron' (layer.py:1225)
                if (prevLayers[d.Inputs[1]].Type != "softmax")
                    throw new LayerParsingException($"Layer '{name}': second input must be softmax layer", name);
                return d;
            }

            default:
                throw new LayerParsingException($"Layer '{name}': Unknown layer type: '{ltype}'", name);
        }
    }

    private static LayerDraft ParseLayerWithInput(
        string name,
        string ltype,
        LayerKind kind,
        IniConfigDocument doc,
        Dictionary<string, LayerDraft> prevLayers,
        int requiredNumInputs)
    {
        var inputs = doc.SafeGetStringList(name, "inputs");
        foreach (var inp in inputs)
        {
            if (!prevLayers.TryGetValue(inp, out var inpDraft))
                throw new LayerParsingException($"Layer '{name}': input layer '{inp}' not defined", name);
            if (inpDraft.Type.StartsWith("cost.", StringComparison.Ordinal))
                throw new LayerParsingException($"Layer '{name}': input layer '{inp}' does not produce any output", name);
        }

        var neuron = doc.SafeGet(name, "neuron", "");
        if (requiredNumInputs > 0 && inputs.Count != requiredNumInputs)
            throw new LayerParsingException($"Layer '{name}': number of inputs must be {requiredNumInputs}", name);

        int gpu = doc.SafeGetInt(name, "gpu", prevLayers[inputs[0]].Gpu);
        return new LayerDraft
        {
            Name = name,
            Type = ltype,
            Kind = kind,
            DeclaredInputs = inputs.ToList(),
            Inputs = inputs.ToList(),
            Neuron = neuron,
            Gpu = gpu,
        };
    }

    private static void ParseWeightLayerBase(
        LayerDraft d, IniConfigDocument doc, Dictionary<string, LayerDraft> prevLayers)
    {
        var name = d.Name;
        int n = d.Inputs.Count;
        d.RequiresParams = true;
        d.InitW = doc.SafeGetFloatListWithScalarDefaultQuirk(name, "initW");
        d.InitB = doc.SafeGetFloat(name, "initB", 0.0);
        d.InitWFunc = doc.SafeGet(name, "initWFunc", "");
        d.InitBFunc = doc.SafeGet(name, "initBFunc", "");
        if (d.InitWFunc.Length > 0)
            throw new LayerParsingException($"Layer '{name}': initWFunc '{d.InitWFunc}' is unsupported in this port", name);
        if (d.InitBFunc.Length > 0)
            throw new LayerParsingException($"Layer '{name}': initBFunc '{d.InitBFunc}' is unsupported in this port", name);

        d.WeightSource = doc.SafeGetStringList(name, "weightSource", Enumerable.Repeat("", n).ToArray());
        NetworkConfigValidators.VerifyNumParams(name, n, "initW", d.InitW.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "weightSource", d.WeightSource.Count);

        var srcLayers = new string[n];
        var srcIndices = new int[n];
        for (int i = 0; i < n; i++)
        {
            var spec = d.WeightSource[i];
            if (spec.Length == 0)
            {
                srcLayers[i] = "";
                srcIndices[i] = -1;
                continue;
            }
            var parsed = NetworkConfigValidators.TryParseWeightSource(spec)
                ?? throw new LayerParsingException(
                    $"Layer '{name}': unable to parse weight sharing source '{spec}'. Format is layer[idx] or just layer, in which case idx=0 is used.",
                    name);
            var (srcName, matIdx) = parsed;
            if (srcName != name && !prevLayers.ContainsKey(srcName))
                throw new LayerParsingException($"Layer '{name}': weight sharing source layer '{srcName}' does not exist.", name);
            var srcLayer = srcName == name ? d : prevLayers[srcName];
            if (srcLayer.Gpu != d.Gpu)
            {
                throw new LayerParsingException(
                    $"Layer '{name}': weight sharing source layer '{srcName}' runs on GPU {srcLayer.Gpu}, while '{name}' runs on GPU {d.Gpu}.",
                    name);
            }
            if (srcLayer.Type != d.Type)
            {
                throw new LayerParsingException(
                    $"Layer '{name}': weight sharing source layer '{srcName}' is of type '{srcLayer.Type}'; should be '{d.Type}'.",
                    name);
            }
            if (srcName != name && srcLayer.Inputs.Count <= matIdx)
            {
                throw new LayerParsingException(
                    $"Layer '{name}': weight sharing source layer '{srcName}' has {srcLayer.Inputs.Count} weight matrices, but '{spec}[{matIdx}]' requested.",
                    name);
            }
            if (srcName == name && matIdx >= i)
                throw new LayerParsingException($"Layer '{name}': weight sharing source '{name}[{matIdx}]' not defined yet.", name);
            srcLayers[i] = srcName;
            srcIndices[i] = matIdx;
        }
        d.WeightSourceLayers = srcLayers;
        d.WeightSourceMatrixIndices = srcIndices;
    }

    private static void ParseConvLayer(
        LayerDraft d, IniConfigDocument doc, Dictionary<string, LayerDraft> prevLayers)
    {
        ParseWeightLayerBase(d, doc, prevLayers);
        var name = d.Name;
        int n = d.Inputs.Count;

        var channels = doc.SafeGetIntList(name, "channels");
        var padding = doc.SafeGetIntList(name, "padding", Enumerable.Repeat(0, n).ToArray());
        var stride = doc.SafeGetIntList(name, "stride", Enumerable.Repeat(1, n).ToArray());
        var filterSize = doc.SafeGetIntList(name, "filterSize");
        var filters = doc.SafeGetIntList(name, "filters");
        var groups = doc.SafeGetIntList(name, "groups", Enumerable.Repeat(1, n).ToArray());
        var randSparse = doc.SafeGetBoolList(name, "randSparse", Enumerable.Repeat(false, n).ToArray());
        var initW = doc.SafeGetFloatList(name, "initW");
        var initCFunc = doc.SafeGet(name, "initCFunc", "");
        if (initCFunc.Length > 0)
            throw new LayerParsingException($"Layer '{name}': initCFunc '{initCFunc}' is unsupported in this port", name);

        NetworkConfigValidators.VerifyNumParams(name, n, "channels", channels.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "padding", padding.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "stride", stride.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "filterSize", filterSize.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "filters", filters.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "groups", groups.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "randSparse", randSparse.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "initW", initW.Count);

        NetworkConfigValidators.VerifyIntListRange(name, stride, "stride", 1, null);
        NetworkConfigValidators.VerifyIntListRange(name, filterSize, "filterSize", 1, null);
        NetworkConfigValidators.VerifyIntListRange(name, padding, "padding", 0, null);
        NetworkConfigValidators.VerifyIntListRange(name, channels, "channels", 1, null);
        NetworkConfigValidators.VerifyIntListRange(name, groups, "groups", 1, null);

        var effFilters = new int[n];
        for (int i = 0; i < n; i++)
            effFilters[i] = filters[i] * groups[i];
        if (effFilters.Distinct().Count() != 1)
            throw new LayerParsingException($"Layer '{name}': all inputs must produce equally-dimensioned output.", name);

        var defaultFc = new int[n];
        for (int i = 0; i < n; i++)
            defaultFc[i] = channels[i] / groups[i];
        IReadOnlyList<int>? explicitFc = null;
        if (randSparse.Any(x => x))
        {
            explicitFc = doc.SafeGetIntList(name, "filterChannels", defaultFc);
            NetworkConfigValidators.VerifyNumParams(name, n, "filterChannels", explicitFc.Count);
        }

        var fc = explicitFc ?? defaultFc;
        for (int i = 0; i < n; i++)
        {
            if (channels[i] > 3 && channels[i] % 4 != 0)
            {
                throw new LayerParsingException(
                    $"Layer '{name}[{i}]': number of channels must be smaller than 4 or divisible by 4", name);
            }
            if (randSparse[i])
            {
                if (groups[i] == 1)
                {
                    throw new LayerParsingException(
                        $"Layer '{name}[{i}]': number of groups must be greater than 1 when using random sparse connectivity", name);
                }
                NetworkConfigValidators.VerifyDivisible(name, true, channels[i], fc[i], "channels", "filterChannels", i);
                NetworkConfigValidators.VerifyDivisible(name, true, fc[i], 4, "filterChannels", null, i);
                NetworkConfigValidators.VerifyDivisible(name, true, groups[i] * fc[i], channels[i], "groups * filterChannels", "channels", i);
            }
            else
            {
                if (groups[i] > 1)
                    NetworkConfigValidators.VerifyDivisible(name, true, channels[i], 4 * groups[i], "channels", "4 * groups", i);
                NetworkConfigValidators.VerifyDivisible(name, true, channels[i], groups[i], "channels", "groups", i);
            }
            NetworkConfigValidators.VerifyDivisible(name, true, effFilters[0], 16 * groups[i], "filters * groups", null, i);
        }

        d.ConvChannels = channels;
        d.ConvFilters = filters;
        d.ConvOutputFilters = effFilters[0];
        d.ConvConfigPadding = padding;
        d.ConvPaddingStart = padding.Select(p => -p).ToArray();
        d.ConvStride = stride;
        d.ConvFilterSize = filterSize;
        d.ConvGroups = groups;
        d.ConvRandSparse = randSparse;
        d.ConvExplicitFilterChannels = explicitFc;
        d.InitW = initW;
        d.InitCFunc = initCFunc;
        d.PartialSum = doc.SafeGetInt(name, "partialSum");
        d.SharedBiases = doc.SafeGetBool(name, "sharedBiases", true);
    }

    private static void ParsePoolLayer(LayerDraft d, IniConfigDocument doc)
    {
        var name = d.Name;
        d.SingleChannels = doc.SafeGetInt(name, "channels");
        d.PoolSizeX = doc.SafeGetInt(name, "sizeX");
        d.PoolStart = doc.SafeGetInt(name, "start", 0);
        d.PoolStride = doc.SafeGetInt(name, "stride");
        d.PoolOutputsX = doc.SafeGetInt(name, "outputsX", 0);
        d.PoolKind = doc.SafeGet(name, "pool");
        d.RequiresParams = d.PoolKind == "rand";

        NetworkConfigValidators.VerifyIntRange(name, d.PoolSizeX, "sizeX", 1, null);
        NetworkConfigValidators.VerifyIntRange(name, d.PoolStride, "stride", 1, d.PoolSizeX);
        NetworkConfigValidators.VerifyIntRange(name, d.PoolOutputsX, "outputsX", 0, null);
        NetworkConfigValidators.VerifyIntRange(name, d.SingleChannels, "channels", 1, null);
        NetworkConfigValidators.VerifyStrIn(name, d.PoolKind, "pool", AllowedPoolKinds);
    }

    private static void ParseNormLayer(LayerDraft d, IniConfigDocument doc, bool isCrossMap)
    {
        var name = d.Name;
        d.RequiresParams = true;
        d.SingleChannels = doc.SafeGetInt(name, "channels");
        d.NormSize = doc.SafeGetInt(name, "size");
        d.NormBlocked = doc.SafeGetBool(name, "blocked", false);

        NetworkConfigValidators.VerifyIntRange(name, d.SingleChannels, "channels", 1, null);
        if (isCrossMap)
        {
            NetworkConfigValidators.VerifyIntRange(name, d.NormSize, "size", 2, d.SingleChannels);
            if (d.SingleChannels % 16 != 0)
            {
                throw new LayerParsingException(
                    $"Layer '{name}': number of channels must be divisible by 16 when using crossMap", name);
            }
        }
        else
        {
            NetworkConfigValidators.VerifyIntRange(name, d.NormSize, "size", 1, null);
            if (d.SingleChannels > 3 && d.SingleChannels % 4 != 0)
            {
                throw new LayerParsingException(
                    $"Layer '{name}': number of channels must be smaller than 4 or divisible by 4", name);
            }
        }
    }

    private static LayerDefinition FinalizeLayer(LayerDraft d, IniConfigDocument paramDoc)
    {
        var name = d.Name;
        var quantF = NetworkConfigValidators.ParseQuantScheme(
            paramDoc, name, d.SourceLayer, "quantF", "forward quantization scheme");
        var quantB = NetworkConfigValidators.ParseQuantScheme(
            paramDoc, name, d.SourceLayer, "quantB", "backward quantization scheme");

        var map = new SortedDictionary<string, ConfigValue>(StringComparer.Ordinal)
        {
            ["gpu"] = ConfigValue.FromInt(d.Gpu),
            ["quantB"] = ConfigValue.FromString(quantB.Raw),
            ["quantF"] = ConfigValue.FromString(quantF.Raw),
        };

        LayerParameters typed;
        switch (d.Kind)
        {
            case LayerKind.Data:
                map["dataIdx"] = ConfigValue.FromInt(d.DataIdx);
                typed = new DataLayerParams(d.DataIdx, d.Gpu, quantF, quantB);
                break;

            case LayerKind.Neuron:
                map["neuron"] = ConfigValue.FromString(d.Neuron);
                typed = new NeuronLayerParams(d.Neuron, UsesActs: true, UsesInputs: false, d.Gpu, quantF, quantB);
                break;

            case LayerKind.Softmax:
                map["neuron"] = ConfigValue.FromString(d.Neuron);
                typed = new SoftmaxLayerParams(d.Neuron, d.Gpu, quantF, quantB);
                break;

            case LayerKind.Hs:
            {
                bool enable = paramDoc.SafeGetBool(name, "enable");
                map["enable"] = ConfigValue.FromBool(enable);
                map["keep"] = ConfigValue.FromFloat(d.Keep);
                map["neuron"] = ConfigValue.FromString(d.Neuron);
                typed = new DropoutLayerParams(d.Keep, enable, d.Neuron, d.Gpu, quantF, quantB);
                break;
            }

            case LayerKind.CostLogreg:
            {
                double coeff = paramDoc.SafeGetFloat(name, "coeff");
                int topk = paramDoc.SafeGetInt(name, "topk", 1);
                map["coeff"] = ConfigValue.FromFloat(coeff);
                map["topk"] = ConfigValue.FromInt(topk);
                typed = new LogregCostLayerParams(coeff, topk, d.Gpu, quantF, quantB);
                break;
            }

            case LayerKind.Pool:
            {
                bool? doMax = d.PoolKind == "rand" ? paramDoc.SafeGetBool(name, "doMax", false) : null;
                map["channels"] = ConfigValue.FromInt(d.SingleChannels);
                map["neuron"] = ConfigValue.FromString(d.Neuron);
                map["outputsX"] = ConfigValue.FromInt(d.PoolOutputsX);
                map["pool"] = ConfigValue.FromString(d.PoolKind);
                map["sizeX"] = ConfigValue.FromInt(d.PoolSizeX);
                map["start"] = ConfigValue.FromInt(d.PoolStart);
                map["stride"] = ConfigValue.FromInt(d.PoolStride);
                if (doMax.HasValue)
                    map["doMax"] = ConfigValue.FromBool(doMax.Value);
                typed = new PoolLayerParams(
                    d.PoolKind, d.SingleChannels, d.PoolSizeX, d.PoolStart, d.PoolStride,
                    d.PoolOutputsX, doMax, d.Neuron, d.Gpu, quantF, quantB);
                break;
            }

            case LayerKind.CmrNorm:
            case LayerKind.RNorm:
                typed = FinalizeNormLayer(d, paramDoc, quantF, quantB, map);
                break;

            case LayerKind.Fc:
            case LayerKind.Conv:
                typed = FinalizeWeightLayer(d, paramDoc, quantF, quantB, map);
                break;

            default:
                throw new InvalidOperationException($"Unhandled LayerKind: {d.Kind}");
        }

        return new LayerDefinition(
            d.Name, d.Type, d.Kind, d.DeclaredInputs.ToArray(), d.Inputs.ToArray(),
            d.SourceLayer, d.DetachedNeuronLayer, d.RequiresParams, typed, map);
    }

    private static LayerParameters FinalizeNormLayer(
        LayerDraft d,
        IniConfigDocument paramDoc,
        ScheduleSpec quantF,
        ScheduleSpec quantB,
        SortedDictionary<string, ConfigValue> map)
    {
        var name = d.Name;
        bool isCrossMap = d.Kind == LayerKind.CmrNorm;
        double rawScale = paramDoc.SafeGetFloat(name, "scale");
        double scale = isCrossMap ? rawScale / d.NormSize : rawScale / ((double)d.NormSize * d.NormSize);
        double pow = paramDoc.SafeGetFloat(name, "pow");

        map["blocked"] = ConfigValue.FromBool(d.NormBlocked);
        map["channels"] = ConfigValue.FromInt(d.SingleChannels);
        map["neuron"] = ConfigValue.FromString(d.Neuron);
        map["pow"] = ConfigValue.FromFloat(pow);
        map["scale"] = ConfigValue.FromFloat(scale);
        map["size"] = ConfigValue.FromInt(d.NormSize);

        if (isCrossMap)
        {
            double minDiv = paramDoc.SafeGetFloat(name, "minDiv", 1.0);
            map["minDiv"] = ConfigValue.FromFloat(minDiv);
            return new CmrNormLayerParams(
                d.SingleChannels, d.NormSize, d.NormBlocked, d.Neuron, d.Gpu,
                rawScale, scale, pow, minDiv, quantF, quantB);
        }

        return new RNormLayerParams(
            d.SingleChannels, d.NormSize, d.NormBlocked, d.Neuron, d.Gpu,
            rawScale, scale, pow, quantF, quantB);
    }

    private static LayerParameters FinalizeWeightLayer(
        LayerDraft d,
        IniConfigDocument paramDoc,
        ScheduleSpec quantF,
        ScheduleSpec quantB,
        SortedDictionary<string, ConfigValue> map)
    {
        var name = d.Name;
        int n = d.Inputs.Count;
        var zeros = Enumerable.Repeat(0.0, n).ToArray();

        var epsW = paramDoc.SafeGetFloatList(name, "epsW");
        double epsB = paramDoc.SafeGetFloat(name, "epsB");
        var momW = paramDoc.SafeGetFloatList(name, "momW");
        double momB = paramDoc.SafeGetFloat(name, "momB");
        double superEps = paramDoc.SafeGetFloat(name, "superEps", 0.0);
        double superMom = paramDoc.SafeGetFloat(name, "superMom", 0.0);
        var wc = paramDoc.SafeGetFloatList(name, "wc", zeros);
        var wball = paramDoc.SafeGetFloatList(name, "wball", zeros);

        NetworkConfigValidators.VerifyNumParams(name, n, "epsW", epsW.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "momW", momW.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "wc", wc.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "wball", wball.Count);

        var schedW = NetworkConfigValidators.ParseScheduleScheme(paramDoc, name, d.SourceLayer, "schedW");
        var schedB = NetworkConfigValidators.ParseScheduleScheme(paramDoc, name, d.SourceLayer, "schedB");
        bool gradConsumer = epsB > 0.0 || epsW.Any(w => w > 0.0);

        map["epsB"] = ConfigValue.FromFloat(epsB);
        map["epsW"] = ConfigValue.FromFloatList(epsW);
        map["initB"] = ConfigValue.FromFloat(d.InitB);
        map["initBFunc"] = ConfigValue.FromString(d.InitBFunc);
        map["initW"] = ConfigValue.FromFloatList(d.InitW);
        map["initWFunc"] = ConfigValue.FromString(d.InitWFunc);
        map["momB"] = ConfigValue.FromFloat(momB);
        map["momW"] = ConfigValue.FromFloatList(momW);
        map["neuron"] = ConfigValue.FromString(d.Neuron);
        map["schedB"] = ConfigValue.FromString(schedB.Raw);
        map["schedW"] = ConfigValue.FromString(schedW.Raw);
        map["superEps"] = ConfigValue.FromFloat(superEps);
        map["superMom"] = ConfigValue.FromFloat(superMom);
        map["wball"] = ConfigValue.FromFloatList(wball);
        map["wc"] = ConfigValue.FromFloatList(wc);
        map["weightSource"] = ConfigValue.FromStringList(d.WeightSource);

        if (d.Kind == LayerKind.Fc)
        {
            map["outputs"] = ConfigValue.FromInt(d.FcOutputs);
            return new FcLayerParams(
                d.FcOutputs, d.InitW, d.InitB, d.InitWFunc, d.InitBFunc,
                d.WeightSource, d.WeightSourceLayers, d.WeightSourceMatrixIndices,
                d.Neuron, d.Gpu, epsW, epsB, momW, momB, superEps, superMom,
                wc, wball, schedW, schedB, quantF, quantB, gradConsumer);
        }

        var wcNormMax = paramDoc.SafeGetFloatList(name, "wcNormMax", zeros);
        var wcNormMin = paramDoc.SafeGetFloatList(name, "wcNormMin", zeros);
        NetworkConfigValidators.VerifyNumParams(name, n, "wcNormMax", wcNormMax.Count);
        NetworkConfigValidators.VerifyNumParams(name, n, "wcNormMin", wcNormMin.Count);
        for (int i = 0; i < n; i++)
        {
            if (wcNormMin[i] > wcNormMax[i])
                throw new LayerParsingException($"Layer '{name}': wcNormMin must be <= wcNormMax.", name);
        }

        map["channels"] = ConfigValue.FromIntList(d.ConvChannels);
        map["filterSize"] = ConfigValue.FromIntList(d.ConvFilterSize);
        map["filters"] = ConfigValue.FromIntList(d.ConvFilters);
        map["groups"] = ConfigValue.FromIntList(d.ConvGroups);
        map["initCFunc"] = ConfigValue.FromString(d.InitCFunc);
        map["padding"] = ConfigValue.FromIntList(d.ConvPaddingStart);
        map["partialSum"] = ConfigValue.FromInt(d.PartialSum);
        map["randSparse"] = ConfigValue.FromBoolList(d.ConvRandSparse);
        map["sharedBiases"] = ConfigValue.FromBool(d.SharedBiases);
        map["stride"] = ConfigValue.FromIntList(d.ConvStride);
        map["wcNormMax"] = ConfigValue.FromFloatList(wcNormMax);
        map["wcNormMin"] = ConfigValue.FromFloatList(wcNormMin);
        if (d.ConvExplicitFilterChannels is not null)
            map["filterChannels"] = ConfigValue.FromIntList(d.ConvExplicitFilterChannels);

        return new ConvLayerParams(
            d.ConvChannels, d.ConvFilters, d.ConvOutputFilters, d.ConvConfigPadding, d.ConvPaddingStart,
            d.ConvStride, d.ConvFilterSize, d.ConvGroups, d.ConvRandSparse, d.ConvExplicitFilterChannels,
            d.InitW, d.InitB, d.InitWFunc, d.InitBFunc, d.InitCFunc,
            d.WeightSource, d.WeightSourceLayers, d.WeightSourceMatrixIndices,
            d.PartialSum, d.SharedBiases, d.Neuron, d.Gpu,
            epsW, epsB, momW, momB, superEps, superMom, wc, wball, wcNormMin, wcNormMax,
            schedW, schedB, quantF, quantB, gradConsumer);
    }
}
