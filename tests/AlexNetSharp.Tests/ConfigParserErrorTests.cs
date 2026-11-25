using AlexNetSharp.Core.Config;

namespace AlexNetSharp.Tests;

internal static class ConfigParserErrorTests
{
    private const string MinimalValidLayers = """
        [data]
        type=data
        dataIdx=0

        [labels]
        type=data
        dataIdx=1

        [fc1]
        type=fc
        inputs=data
        outputs=10
        initW=0.01
        gpu=0

        [probs]
        type=softmax
        inputs=fc1

        [logprob]
        type=cost.logreg
        inputs=labels,probs
        """;

    private const string MinimalValidParams = """
        [fc1]
        epsW=0.01
        epsB=0.02
        momW=0.9
        momB=0.9

        [logprob]
        coeff=1
        """;

    public static int RunAll()
    {
        var tests = new Action[]
        {
            TestMissingType,
            TestUnknownTypeAndUnsupportedPortTypes,
            TestInputNotDefinedEarlier,
            TestUnusedLayer,
            TestMissingRequiredParameter,
            TestBadInt,
            TestBadFloat,
            TestBadBoolAndAllValidBoolTokens,
            TestListLengthMismatch,
            TestMissingParamsSectionForLayerRequiringParams,
            TestOptionNameCaseFoldingAndSectionCaseSensitivity,
            TestSafeConfigParserInterpolationAndDefaults,
            TestCommentsColonDelimitersAndContinuations,
        };

        foreach (var test in tests)
            test();

        return tests.Length;
    }

    private static void TestMissingType()
    {
        var cfg = """
            [broken_sec]
            dataIdx=0
            """;
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(cfg, ""),
            "Layer 'broken_sec': no type given",
            nameof(TestMissingType));
    }

    private static void TestUnknownTypeAndUnsupportedPortTypes()
    {
        var unknownCfg = """
            [mystery]
            type=quantum_conv
            """;
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(unknownCfg, ""),
            "Layer 'mystery': Unknown layer type: 'quantum_conv'",
            "unknown layer type");

        var unsupportedLayerCfg = """
            [loc1]
            type=local
            """;
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(unsupportedLayerCfg, ""),
            "Layer 'loc1': layer type 'local' is unsupported in this port",
            "unsupported known layer type");

        var unsupportedNeuronCfg = MinimalValidLayers.Replace("outputs=10", "outputs=10\nneuron=tanh[1.0,0.5]");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(unsupportedNeuronCfg, MinimalValidParams),
            "Layer 'fc1': neuron type 'tanh[1.0,0.5]' is unsupported in this port",
            "unsupported known neuron type");

        var badNeuronCfg = MinimalValidLayers.Replace("outputs=10", "outputs=10\nneuron=swish");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(badNeuronCfg, MinimalValidParams),
            "Layer 'fc1': unable to parse neuron type 'swish'.",
            "unknown neuron type");
    }

    private static void TestInputNotDefinedEarlier()
    {
        // Forward reference: probs appears before fc1
        var forwardRefCfg = """
            [data]
            type=data
            dataIdx=0

            [labels]
            type=data
            dataIdx=1

            [probs]
            type=softmax
            inputs=fc1

            [fc1]
            type=fc
            inputs=data
            outputs=10
            initW=0.01
            gpu=0

            [logprob]
            type=cost.logreg
            inputs=labels,probs
            """;
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(forwardRefCfg, MinimalValidParams),
            "Layer 'probs': input layer 'fc1' not defined",
            "forward reference input");
    }

    private static void TestUnusedLayer()
    {
        var unusedCfg = MinimalValidLayers + "\n[dangling_pool]\ntype=pool\npool=max\ninputs=data\nsizeX=3\nstride=2\nchannels=3\n";
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(unusedCfg, MinimalValidParams),
            "Layer 'dangling_pool' of type 'pool' is unused",
            nameof(TestUnusedLayer));
    }

    private static void TestMissingRequiredParameter()
    {
        var missingCfgParam = MinimalValidLayers.Replace("outputs=10\n", "");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(missingCfgParam, MinimalValidParams),
            "Layer 'fc1': required parameter 'outputs' missing",
            "missing required cfg param");

        var missingParamsFileOpt = MinimalValidParams.Replace("epsB=0.02\n", "");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(MinimalValidLayers, missingParamsFileOpt),
            "Layer 'fc1': required parameter 'epsB' missing",
            "missing required params-file param");
    }

    private static void TestBadInt()
    {
        var badScalarInt = MinimalValidLayers.Replace("outputs=10", "outputs=10.5");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(badScalarInt, MinimalValidParams),
            "Layer 'fc1': parameter 'outputs' must be int",
            "bad scalar int");

        var badListInt = """
            [data]
            type=data
            dataIdx=0

            [conv1]
            type=conv
            inputs=data
            channels=three
            filters=48
            filterSize=11
            initW=0.01
            partialSum=1
            gpu=0
            """;
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(badListInt, ""),
            "Layer 'conv1': parameter 'channels' must be ','-delimited list of ints",
            "bad int list");
    }

    private static void TestBadFloat()
    {
        var badScalarFloat = MinimalValidParams.Replace("epsB=0.02", "epsB=not_a_float");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(MinimalValidLayers, badScalarFloat),
            "Layer 'fc1': parameter 'epsB' must be float",
            "bad scalar float");

        var badListFloat = MinimalValidParams.Replace("epsW=0.01", "epsW=0.01,xyz");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(MinimalValidLayers, badListFloat),
            "Layer 'fc1': parameter 'epsW' must be ','-delimited list of floats",
            "bad float list");
    }

    private static void TestBadBoolAndAllValidBoolTokens()
    {
        var hsLayers = MinimalValidLayers.Replace(
            "inputs=fc1",
            "inputs=hs1\n\n[hs1]\ntype=hs\ninputs=fc1\nkeep=0.5");
        // Fix section order so hs1 is before probs
        hsLayers = """
            [data]
            type=data
            dataIdx=0

            [labels]
            type=data
            dataIdx=1

            [fc1]
            type=fc
            inputs=data
            outputs=10
            initW=0.01
            gpu=0

            [hs1]
            type=hs
            inputs=fc1
            keep=0.5

            [probs]
            type=softmax
            inputs=hs1

            [logprob]
            type=cost.logreg
            inputs=labels,probs
            """;

        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(hsLayers, MinimalValidParams + "\n[hs1]\nenable=maybe\n"),
            "Layer 'hs1': parameter 'enable' must be bool",
            "bad bool");

        foreach (var trueTok in new[] { "1", "yes", "YES", "true", "True", "on", "ON" })
        {
            var net = NetworkConfigParser.Parse(hsLayers, MinimalValidParams + $"\n[hs1]\nenable={trueTok}\n");
            TestAssert.Equal(true, ((DropoutLayerParams)net["hs1"].TypedParams).Enable, $"bool true token '{trueTok}'");
        }
        foreach (var falseTok in new[] { "0", "no", "NO", "false", "False", "off", "OFF" })
        {
            var net = NetworkConfigParser.Parse(hsLayers, MinimalValidParams + $"\n[hs1]\nenable={falseTok}\n");
            TestAssert.Equal(false, ((DropoutLayerParams)net["hs1"].TypedParams).Enable, $"bool false token '{falseTok}'");
        }
    }

    private static void TestListLengthMismatch()
    {
        var cfgMismatch = MinimalValidLayers.Replace("initW=0.01", "initW=0.01,0.02");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(cfgMismatch, MinimalValidParams),
            "Layer 'fc1': initW list length does not match number of inputs",
            "cfg verify_num_params");

        var paramMismatch = MinimalValidParams.Replace("epsW=0.01", "epsW=0.01,0.01");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(MinimalValidLayers, paramMismatch),
            "Layer 'fc1': epsW list length does not match number of inputs",
            "params verify_num_params");
    }

    private static void TestMissingParamsSectionForLayerRequiringParams()
    {
        var paramsWithoutFc1 = """
            [logprob]
            coeff=1
            """;
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(MinimalValidLayers, paramsWithoutFc1, "layers.cfg", "layer-params.cfg"),
            "Layer 'fc1' of type 'fc' requires extra parameters, but none given in file 'layer-params.cfg'.",
            nameof(TestMissingParamsSectionForLayerRequiringParams));
    }

    private static void TestOptionNameCaseFoldingAndSectionCaseSensitivity()
    {
        var upperCaseOptionsParams = """
            [fc1]
            EPSW=0.025
            EpsB=0.05
            MOMW=0.85
            MomB=0.85

            [logprob]
            COEFF=2.5
            TOPK=3
            """;
        var net = NetworkConfigParser.Parse(MinimalValidLayers, upperCaseOptionsParams);
        var fc1 = (FcLayerParams)net["fc1"].TypedParams;
        TestAssert.SequenceEqual([0.025], fc1.EpsW, "EPSW case-folded to epsW");
        TestAssert.FloatEqual(0.05, fc1.EpsB, "EpsB case-folded to epsB");
        TestAssert.Equal(3, ((LogregCostLayerParams)net["logprob"].TypedParams).TopK, "TOPK case-folded to topk");

        // Section names ARE case-sensitive: [FC1] does not satisfy [fc1]
        var wrongCaseSectionParams = upperCaseOptionsParams.Replace("[fc1]", "[FC1]");
        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.Parse(MinimalValidLayers, wrongCaseSectionParams, "l.cfg", "p.cfg"),
            "Layer 'fc1' of type 'fc' requires extra parameters, but none given in file 'p.cfg'.",
            "section names are case-sensitive");
    }

    private static void TestSafeConfigParserInterpolationAndDefaults()
    {
        var interpParams = """
            [DEFAULT]
            base_lr = 0.001
            pair_lr = %(base_lr)s, %(base_lr)s
            mom = 0.9
            momB = %(mom)s

            [fc1]
            epsW = %(BASE_LR)s
            epsB = 0.02
            momW = %(mom)s

            [fc2]
            base_lr = 0.0005
            epsW = %(pair_lr)s
            epsB = 0.02
            momW = %(mom)s, %(mom)s

            [logprob]
            coeff = 1
            """;
        var twoFcLayers = """
            [data]
            type=data
            dataIdx=0

            [labels]
            type=data
            dataIdx=1

            [fc1]
            type=fc
            inputs=data
            outputs=10
            initW=0.01
            gpu=0

            [fc2]
            type=fc
            inputs=data,fc1
            outputs=10
            initW=0.01,0.01

            [probs]
            type=softmax
            inputs=fc2

            [logprob]
            type=cost.logreg
            inputs=labels,probs
            """;

        var net = NetworkConfigParser.Parse(twoFcLayers, interpParams);
        TestAssert.SequenceEqual([0.001], ((FcLayerParams)net["fc1"].TypedParams).EpsW, "fc1 interpolated DEFAULT base_lr");
        TestAssert.SequenceEqual([0.0005, 0.0005], ((FcLayerParams)net["fc2"].TypedParams).EpsW, "fc2 overrode base_lr in recursive pair_lr");
        TestAssert.FloatEqual(0.9, ((FcLayerParams)net["fc1"].TypedParams).MomB, "fc1 inherited DEFAULT momB");

        // Escaped %% and interpolation error cases
        var doc = IniConfigDocument.Parse("[DEFAULT]\npct = 50%%\n[s]\nlabel = rate=%(pct)s\n");
        TestAssert.Equal("rate=50%", doc.SafeGet("s", "label"), "escaped %% inside interpolation");

        var missingRefDoc = IniConfigDocument.Parse("[s]\nx = %(missing_key)s\n");
        TestAssert.ThrowsLayerError(
            () => missingRefDoc.SafeGet("s", "x"),
            "interpolation option 'missing_key' not found",
            "missing interpolation option");

        var cycleDoc = IniConfigDocument.Parse("[s]\na = %(b)s\nb = %(a)s\n");
        TestAssert.ThrowsLayerError(
            () => cycleDoc.SafeGet("s", "a"),
            "interpolation depth exceeded",
            "cyclic interpolation depth");
    }

    private static void TestCommentsColonDelimitersAndContinuations()
    {
        var fancyLayers = """
            # Hash comment before sections
            ; Semicolon comment before sections
            [data]
            type: data ; inline semicolon comment
            dataIdx: 0

            [labels]
            type = data
            dataIdx = 1

            [fc1]
            type: fc
            inputs: data
            outputs: 10
            initW: 0.01
            gpu: 0

            [fc2]
            type = fc
            inputs:
                data,
                # indented comment inside continuation
                fc1
            outputs = 10
            initW = 0.01,
                0.02

            [probs]
            type: softmax
            inputs = fc2

            [logprob]
            type = cost.logreg
            inputs: labels,
                probs
            """;

        var fancyParams = """
            ; Params with colon delimiters and continuations
            [fc1]
            epsW: 0.01
            epsB: 0.02
            momW: 0.9
            momB: 0.9

            [fc2]
            epsW: 0.001,
                  0.002 ; inline comment on continuation
            epsB = 0.02
            momW = 0.9,
                   0.85
            momB: 0.9

            [logprob]
            coeff: 1.0
            """;

        var net = NetworkConfigParser.Parse(fancyLayers, fancyParams);
        TestAssert.SequenceEqual(["data", "fc1"], net["fc2"].Inputs, "fc2 multiline continuation inputs");
        TestAssert.SequenceEqual([0.01, 0.02], ((FcLayerParams)net["fc2"].TypedParams).InitW, "fc2 multiline initW");
        TestAssert.SequenceEqual([0.001, 0.002], ((FcLayerParams)net["fc2"].TypedParams).EpsW, "fc2 multiline epsW with inline comment");
        TestAssert.SequenceEqual([0.9, 0.85], ((FcLayerParams)net["fc2"].TypedParams).MomW, "fc2 multiline momW");
    }
}
