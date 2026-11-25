using System.Globalization;
using AlexNetSharp.Core.Config;

namespace AlexNetSharp.Tests;

internal static class GoldenParseTests
{
    public static void WriteGoldenFixture(string origRoot, string fixturePath)
    {
        var net = Parse145(origRoot);
        var json = NetworkDefinitionJson.Serialize(net);
        Directory.CreateDirectory(Path.GetDirectoryName(fixturePath)!);
        File.WriteAllText(fixturePath, json);
    }

    public static int RunAll(string repoRoot, string origRoot)
    {
        int passed = 0;

        TestGoldenFixtureAndCultureInvariance(repoRoot, origRoot);
        passed++;

        TestEightHandCheckedLayers(origRoot);
        passed++;

        TestSmokeParseOtherOriginalConfigs(origRoot);
        passed++;

        return passed;
    }

    private static NetworkDefinition Parse145(string origRoot) =>
        NetworkConfigParser.ParseFiles(
            Path.Combine(origRoot, "layers", "layers-145.cfg"),
            Path.Combine(origRoot, "layers", "layer-params-145.cfg"));

    private static void TestGoldenFixtureAndCultureInvariance(string repoRoot, string origRoot)
    {
        var fixturePath = Path.Combine(repoRoot, "tests", "fixtures", "layers-145.parsed.json");
        TestAssert.True(File.Exists(fixturePath), $"Missing golden fixture: {fixturePath}");

        var expectedJson = File.ReadAllText(fixturePath);
        TestAssert.True(!expectedJson.Contains('\r'), "Golden fixture must use LF line endings only.");
        TestAssert.True(!expectedJson.Contains(origRoot, StringComparison.OrdinalIgnoreCase),
            "Golden fixture must never embed ALEXNET_ORIGINAL_SRC path.");

        var savedCulture = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("fr-FR"); // Decimal comma culture
            var net = Parse145(origRoot);
            var actualJson = NetworkDefinitionJson.Serialize(net);
            TestAssert.Equal(expectedJson, actualJson, "Golden parse JSON parity under fr-FR culture");
        }
        finally
        {
            CultureInfo.CurrentCulture = savedCulture;
        }
    }

    private static void TestEightHandCheckedLayers(string origRoot)
    {
        var net = Parse145(origRoot);
        TestAssert.Equal(49, net.Layers.Count, "total layers (35 declared + 14 detached)");
        TestAssert.Equal(35, net.DeclaredLayers.Count, "declared layers");
        TestAssert.Equal(14, net.DetachedNeuronLayers.Count, "detached neuron layers");

        // 1. conv1a: frozen epsW=0, no ReLU, padding=0, sharedBiases=true, default initB=0
        var conv1a = net["conv1a"];
        var conv1aP = (ConvLayerParams)conv1a.TypedParams;
        TestAssert.SequenceEqual(["data"], conv1a.Inputs, "conv1a.Inputs");
        TestAssert.Equal((string?)null, conv1a.DetachedNeuronLayer, "conv1a.DetachedNeuronLayer");
        TestAssert.SequenceEqual([3], conv1aP.Channels, "conv1a.Channels");
        TestAssert.SequenceEqual([48], conv1aP.Filters, "conv1a.Filters");
        TestAssert.SequenceEqual([0], conv1aP.PaddingStart, "conv1a.PaddingStart");
        TestAssert.SequenceEqual([4], conv1aP.Stride, "conv1a.Stride");
        TestAssert.SequenceEqual([11], conv1aP.FilterSize, "conv1a.FilterSize");
        TestAssert.FloatEqual(0.0, conv1aP.InitB, "conv1a.InitB");
        TestAssert.SequenceEqual([0.0], conv1aP.EpsW, "conv1a.EpsW");
        TestAssert.Equal(false, conv1aP.GradConsumer, "conv1a.GradConsumer");

        // 2. conv3a: two inputs (pool2a,pool2b), initW=0.03,0.03, initB=0, padding=1,1 -> -1,-1, detached relu
        var conv3a = net["conv3a"];
        var conv3aP = (ConvLayerParams)conv3a.TypedParams;
        TestAssert.SequenceEqual(["pool2a", "pool2b"], conv3a.Inputs, "conv3a.Inputs");
        TestAssert.Equal("conv3a_neuron", conv3a.DetachedNeuronLayer, "conv3a.DetachedNeuronLayer");
        TestAssert.SequenceEqual([1, 1], conv3aP.ConfigPadding, "conv3a.ConfigPadding");
        TestAssert.SequenceEqual([-1, -1], conv3aP.PaddingStart, "conv3a.PaddingStart");
        TestAssert.SequenceEqual([0.03, 0.03], conv3aP.InitW, "conv3a.InitW");
        TestAssert.FloatEqual(0.0, conv3aP.InitB, "conv3a.InitB");
        TestAssert.SequenceEqual([1e-5, 1e-5], conv3aP.EpsW, "conv3a.EpsW");

        // 3. rnorm2a (cmrnorm): input rewritten to conv2a_neuron, scale=0.0001/5=2e-5, minDiv=2
        var rnorm2a = net["rnorm2a"];
        var rnorm2aP = (CmrNormLayerParams)rnorm2a.TypedParams;
        TestAssert.SequenceEqual(["conv2a"], rnorm2a.DeclaredInputs, "rnorm2a.DeclaredInputs");
        TestAssert.SequenceEqual(["conv2a_neuron"], rnorm2a.Inputs, "rnorm2a.Inputs");
        TestAssert.FloatEqual(0.0001, rnorm2aP.RawScale, "rnorm2a.RawScale");
        TestAssert.FloatEqual(0.00002, rnorm2aP.Scale, "rnorm2a.Scale");
        TestAssert.FloatEqual(2.0, rnorm2aP.MinDiv, "rnorm2a.MinDiv");

        // 4. cnorm2a (rnorm): spatial norm, scale=0.001/25=4e-5, no minDiv
        var cnorm2a = net["cnorm2a"];
        var cnorm2aP = (RNormLayerParams)cnorm2a.TypedParams;
        TestAssert.SequenceEqual(["rnorm2a"], cnorm2a.Inputs, "cnorm2a.Inputs");
        TestAssert.FloatEqual(0.00004, cnorm2aP.Scale, "cnorm2a.Scale");
        TestAssert.True(!cnorm2a.Parameters.ContainsKey("minDiv"), "cnorm2a must not emit minDiv");

        // 5. hs1a: input rewritten to fc2048a_neuron, keep=0.5, enable=true, gpu=0 inherited
        var hs1a = net["hs1a"];
        var hs1aP = (DropoutLayerParams)hs1a.TypedParams;
        TestAssert.SequenceEqual(["fc2048a_neuron"], hs1a.Inputs, "hs1a.Inputs");
        TestAssert.FloatEqual(0.5, hs1aP.Keep, "hs1a.Keep");
        TestAssert.Equal(true, hs1aP.Enable, "hs1a.Enable");
        TestAssert.Equal(0, hs1aP.Gpu, "hs1a.Gpu");

        // 6. fc2048bb: reversed input order hs1b,hs1a, outputs=2048, initB=1, gpu=1
        var fc2048bb = net["fc2048bb"];
        var fc2048bbP = (FcLayerParams)fc2048bb.TypedParams;
        TestAssert.SequenceEqual(["hs1b", "hs1a"], fc2048bb.Inputs, "fc2048bb.Inputs");
        TestAssert.Equal(2048, fc2048bbP.Outputs, "fc2048bb.Outputs");
        TestAssert.FloatEqual(1.0, fc2048bbP.InitB, "fc2048bb.InitB");
        TestAssert.Equal(1, fc2048bbP.Gpu, "fc2048bb.Gpu");

        // 7. fc1000: inputs hs2a,hs2b, outputs=1000, default initB=0, no detached neuron
        var fc1000 = net["fc1000"];
        var fc1000P = (FcLayerParams)fc1000.TypedParams;
        TestAssert.SequenceEqual(["hs2a", "hs2b"], fc1000.Inputs, "fc1000.Inputs");
        TestAssert.Equal(1000, fc1000P.Outputs, "fc1000.Outputs");
        TestAssert.FloatEqual(0.0, fc1000P.InitB, "fc1000.InitB");
        TestAssert.Equal((string?)null, fc1000.DetachedNeuronLayer, "fc1000.DetachedNeuronLayer");

        // 8. logprob: cost.logreg, inputs labels,probs, coeff=1, topk=5, no 'neuron' key
        var logprob = net["logprob"];
        var logprobP = (LogregCostLayerParams)logprob.TypedParams;
        TestAssert.SequenceEqual(["labels", "probs"], logprob.Inputs, "logprob.Inputs");
        TestAssert.FloatEqual(1.0, logprobP.Coeff, "logprob.Coeff");
        TestAssert.Equal(5, logprobP.TopK, "logprob.TopK");
        TestAssert.True(!logprob.Parameters.ContainsKey("neuron"), "logprob must not emit neuron");
    }

    private static void TestSmokeParseOtherOriginalConfigs(string origRoot)
    {
        var net153 = NetworkConfigParser.ParseFiles(
            Path.Combine(origRoot, "layers", "layers-153.cfg"),
            Path.Combine(origRoot, "layers", "layer-params-153.cfg"));
        TestAssert.Equal(35, net153.DeclaredLayers.Count, "layers-153 declared count");

        TestAssert.ThrowsLayerError(
            () => NetworkConfigParser.ParseFiles(
                Path.Combine(origRoot, "layers", "layers-137-tree.cfg"),
                Path.Combine(origRoot, "layers", "layer-params-137-tree.cfg")),
            "is unsupported in this port",
            "layers-137-tree unsupported type");
    }
}
