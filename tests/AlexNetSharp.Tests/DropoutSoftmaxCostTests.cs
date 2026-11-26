using System.Globalization;
using System.Text.Json.Nodes;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Layers.Cost;
using AlexNetSharp.Core.Layers.Dropout;
using AlexNetSharp.Core.Layers.FullyConnected;
using AlexNetSharp.Core.Layers.Softmax;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class DropoutSoftmaxCostTests
{
    internal static FloatMatrix C(int features, int cases) => new(features, cases);
    internal static FloatMatrix S(int cases, int classes) => C(classes, cases).TransposeView();
    internal static float[] Values(FloatMatrix canonical) => Enumerable.Range(0, canonical.Rows)
        .SelectMany(f => Enumerable.Range(0, canonical.Cols).Select(n => canonical[f, n])).ToArray();
    internal static void Bits(IReadOnlyList<float> expected, IReadOnlyList<float> actual, string label)
    {
        TestAssert.Equal(expected.Count, actual.Count, label + " length");
        for (int i = 0; i < expected.Count; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                throw new TestFailureException($"{label}[{i}]: bits {expected[i]:R} != {actual[i]:R}");
    }
    internal static void Bits(FloatMatrix expected, FloatMatrix actual, string label) => Bits(Values(expected), Values(actual), label);
    private static double D(JsonNode node) => node is JsonValue v && v.TryGetValue<string>(out var s) ?
        double.Parse(s, CultureInfo.InvariantCulture) : node.GetValue<double>();
    private static float F(JsonNode node) => (float)D(node);
    private static FloatMatrix Read(int features, int cases, JsonNode node)
    {
        var m = C(features, cases); var a = node.AsArray(); TestAssert.Equal(m.ElementCount, a.Count, "fixture matrix length");
        for (int i = 0; i < a.Count; i++) m[i / cases, i % cases] = F(a[i]!);
        return m;
    }
    private static void Near(double e, double a, string label, double tolerance = 2e-6)
    {
        bool ok = double.IsNaN(e) ? double.IsNaN(a) : double.IsInfinity(e) ? e == a : double.IsFinite(a) && Math.Abs(e - a) <= tolerance * (1 + Math.Abs(e));
        TestAssert.True(ok, $"{label}: expected {e:R}, actual {a:R}");
    }
    private static void Check(JsonNode expected, IReadOnlyList<float> actual, string label, double tolerance = 2e-6, float offset = 0)
    {
        var a = expected.AsArray(); TestAssert.Equal(a.Count, actual.Count, label + " count");
        for (int i = 0; i < a.Count; i++) Near(D(a[i]!) + offset, actual[i], label + $"[{i}]", tolerance);
    }
    private static void CheckCost(JsonNode expected, LogregCostResult actual)
    {
        Check(expected["logs"]!, actual.LabelLogProbabilities, "cost logs");
        Check(expected["correct"]!, actual.Top1Correct, "top1"); Check(expected["top"]!, actual.TopKCorrect, "topk");
        Near(D(expected["totals"]![0]!), actual.NegativeLogLikelihood, "cost sum");
        Near(D(expected["totals"]![1]!), actual.Top1Error, "top1 sum"); Near(D(expected["totals"]![2]!), actual.TopKError, "topk sum");
    }
    internal static void Bits(LogregCostResult e, LogregCostResult a)
    {
        TestAssert.Equal(BitConverter.DoubleToInt64Bits(e.NegativeLogLikelihood), BitConverter.DoubleToInt64Bits(a.NegativeLogLikelihood), "loss bits");
        TestAssert.Equal(BitConverter.DoubleToInt64Bits(e.Top1Error), BitConverter.DoubleToInt64Bits(a.Top1Error), "top1 bits");
        TestAssert.Equal(BitConverter.DoubleToInt64Bits(e.TopKError), BitConverter.DoubleToInt64Bits(a.TopKError), "topk bits");
        Bits(e.LabelLogProbabilities, a.LabelLogProbabilities, "logs"); Bits(e.Top1Correct, a.Top1Correct, "top1"); Bits(e.TopKCorrect, a.TopKCorrect, "topk");
    }
    internal static void Fill(FloatMatrix m, Random rng)
    {
        for (int f = 0; f < m.Rows; f++) for (int n = 0; n < m.Cols; n++) m[f, n] = (float)(rng.NextDouble() * 2 - 1);
    }
    private static void Reject(Action action)
    {
        try { action(); } catch (Exception ex) when (ex is ArgumentException or InvalidOperationException) { return; }
        throw new TestFailureException("Expected contract rejection.");
    }
    public static int RunAll(string repo, string original, bool full)
    {
        var root = JsonNode.Parse(File.ReadAllText(Path.Combine(repo, "tests/fixtures/dropout-softmax-cost-cases.json")))!;
        int fixtures = Fixtures(root), parity = Parity(), gradients = Numerical(), contracts = Contracts(), real = Real(original, full);
        Console.WriteLine($"DROPOUT/SOFTMAX/COST TESTS PASSED: {fixtures} fixture, {parity} bit-parity/RNG, {gradients} numerical, {contracts} contract, {real} real-geometry groups.");
        return fixtures + parity + gradients + contracts + real;
    }
    private static int Fixtures(JsonNode root)
    {
        int count = 0;
        foreach (var f in root["dropout"]!.AsArray().Select(x => x!))
        {
            int features = f["features"]!.GetValue<int>(), cases = f["cases"]!.GetValue<int>();
            var x = Read(features, cases, f["inputs"]!); var dy = Read(features, cases, f["upstream"]!);
            var mask = f["mask"]!.AsArray().Select(v => F(v!)).ToArray(); var mode = Enum.Parse<DropoutMode>(f["mode"]!.GetValue<string>());
            foreach (bool fast in new[] { false, true })
            {
                var e = new DropoutAdapter(fast, F(f["keep"]!), f["enable"]!.GetValue<bool>(), new FixedDropoutMaskSource(mask));
                var y = C(features, cases); y.Fill(float.NaN); e.Forward(x, y, mode);
                Check(f["output"]!, Values(y), "dropout forward", 0);
                var dx = C(features, cases); dx.Fill(float.NaN); e.Backward(dy, dx);
                Check(f["gradient"]!, Values(dx), "dropout backward", 0);
                dx.Fill(2); e.Backward(dy, dx, 1); Check(f["gradient"]!, Values(dx), "dropout add", 0, 2);
                dx.Fill(float.NaN); e.Backward(dy, dx, 1); TestAssert.True(Values(dx).All(float.IsNaN), "dropout preserves NaN target with scale 1");
                count++;
            }
        }
        foreach (var f in root["softmax"]!.AsArray().Select(x => x!))
        {
            int classes = f["classes"]!.GetValue<int>(), cases = f["cases"]!.GetValue<int>();
            var x = Read(classes, cases, f["logits"]!).TransposeView(); var dy = Read(classes, cases, f["upstream"]!).TransposeView();
            foreach (bool fast in new[] { false, true })
            {
                var y = S(cases, classes); y.Fill(float.NaN); Forward(fast, x, y);
                Check(f["output"]!, Values(y.TransposeView()), "softmax");
                var dx = S(cases, classes); dx.Fill(float.NaN); Backward(fast, y, dy, dx);
                Check(f["gradient"]!, Values(dx.TransposeView()), "softmax general");
                dx.Fill(2); Backward(fast, y, dy, dx, 1); Check(f["gradient"]!, Values(dx.TransposeView()), "softmax add", offset: 2);
                dx.Fill(float.NaN); Backward(fast, y, dy, dx, 1); TestAssert.True(Values(dx).All(float.IsNaN), "softmax NaN target add"); count++;
            }
        }
        foreach (var f in root["fused"]!.AsArray().Select(x => x!))
        {
            int classes = f["classes"]!.GetValue<int>(), cases = f["cases"]!.GetValue<int>();
            var p = Read(classes, cases, f["probabilities"]!); var labels = Read(1, cases, f["labels"]!); float coeff = F(f["coefficient"]!);
            foreach (bool fast in new[] { false, true })
            {
                var dx = S(cases, classes); dx.Fill(float.NaN); Fused(fast, labels, p.TransposeView(), dx, coeff);
                Check(f["gradient"]!, Values(dx.TransposeView()), "fused", 0);
                var dy = C(classes, cases); dy.Fill(float.NaN); ProbabilityBackward(fast, labels, p, dy, coeff);
                Check(f["probability_gradient"]!, Values(dy), "probability gradient");
                var general = S(cases, classes); Backward(fast, p.TransposeView(), dy.TransposeView(), general);
                if (f["name"]!.GetValue<string>() == "regular") Check(f["gradient"]!, Values(general.TransposeView()), "fused/general agreement");
                else TestAssert.True(Values(general).All(float.IsNaN) && Values(dx).All(float.IsFinite), "fusion avoids zero divisions");
                dx.Fill(2); Fused(fast, labels, p.TransposeView(), dx, coeff, 1); Check(f["gradient"]!, Values(dx.TransposeView()), "fused add", offset: 2);
                dx.Fill(float.NaN); Fused(fast, labels, p.TransposeView(), dx, coeff, 1); TestAssert.True(Values(dx).All(float.IsNaN), "fused NaN add");
                dy.Fill(float.NaN); ProbabilityBackward(fast, labels, p, dy, coeff, 1); TestAssert.True(Values(dy).All(float.IsNaN), "cost backward NaN add"); count++;
            }
        }
        foreach (var f in root["cost"]!.AsArray().Select(x => x!))
        {
            int classes = f["classes"]!.GetValue<int>(), cases = f["cases"]!.GetValue<int>();
            var p = Read(classes, cases, f["probabilities"]!); var labels = Read(1, cases, f["labels"]!); int k = f["topk"]!.GetValue<int>();
            foreach (bool fast in new[] { false, true }) { CheckCost(f, Cost(fast, labels, p, k)); count++; }
        }
        foreach (var f in root["multiview"]!.AsArray().Select(x => x!))
        {
            int classes = f["classes"]!.GetValue<int>(), cases = f["cases"]!.GetValue<int>(), k = f["topk"]!.GetValue<int>();
            int views = f["count"]!.GetValue<int>(); var labels = Read(1, cases, f["labels"]!);
            foreach (bool fast in new[] { false, true })
            {
                var a = new LogregMultiviewAccumulator(fast); LogregCostResult? result = null;
                for (int v = 0; v < views; v++)
                {
                    var p = Read(classes, cases, f["views"]![v]!); var saved = Values(p);
                    result = a.Forward(labels, p, k, v == 0 ? MultiviewPhase.Start : v == views - 1 ? MultiviewPhase.End : MultiviewPhase.Continue);
                    if (v < views - 1) TestAssert.True(result == null && a.LastCost == null, "cost deferred until end");
                    Bits(saved, Values(p), "multiview input unchanged");
                }
                Check(f["average"]!, Values(a.CopyProbabilities()), "probability average"); CheckCost(f, result!);
                TestAssert.Equal(views, a.ViewCount, "view count"); var last = a.LastCost;
                var first = Read(classes, cases, f["views"]![0]!);
                a.Forward(labels, first, k, MultiviewPhase.Start);
                TestAssert.True(ReferenceEquals(last, a.LastCost), "pending sequence retains prior cost");
                Bits(first, a.CopyProbabilities(), "start overwrites prior average");
                var one = a.Forward(labels, first, k, MultiviewPhase.Start | MultiviewPhase.End)!;
                Bits(Cost(fast, labels, first, k), one); TestAssert.Equal(1, a.ViewCount, "single view reset"); count++;
            }
        }
        foreach (var f in root["rng"]!.AsArray().Select(x => x!))
        {
            ulong seed = ulong.Parse(f["seed"]!.GetValue<string>()), offset = ulong.Parse(f["offset"]!.GetValue<string>());
            var source = new SeededDropoutMaskSource(seed, offset); var m = C(3, 4); source.Fill(m, .5f, 24);
            Check(f["mask"]!, Values(m), "independent RNG mask", 0);
            for (int i = 0; i < 12; i++) Near(D(f["uniform"]![i]!), SeededDropoutMaskSource.Uniform(seed, unchecked(offset + (ulong)i)), "RNG uniform", 0);
            TestAssert.Equal(unchecked(offset + 12), source.DrawOffset, "RNG cursor");
            if (D(f["uniform"]![0]!) == 1)
            {
                var endpoint = C(1, 1); new SeededDropoutMaskSource(seed, offset).Fill(endpoint, 1);
                TestAssert.Equal(0f, endpoint[0, 0], "u=1 is dropped even with keep=1");
            }
            count++;
        }
        var witness = root["rounding_witness"]!;
        var wp = Read(3, 1, witness["probabilities"]!).TransposeView(); var wu = Read(3, 1, witness["upstream"]!).TransposeView();
        foreach (bool fast in new[] { false, true })
        {
            var dx = S(1, 3); Backward(fast, wp, wu, dx);
            Check(witness["gradient"]!, Values(dx.TransposeView()), "historical j-loop rounding", 0); count++;
        }
        return count;
    }
    internal static void Forward(bool fast, FloatMatrix x, FloatMatrix y, int threads = -1)
    { if (fast) OptimizedSoftmax.Forward(x, y, threads); else ScalarSoftmax.Forward(x, y); }
    internal static void Backward(bool fast, FloatMatrix p, FloatMatrix dy, FloatMatrix dx, float scale = 0, int threads = -1)
    { if (fast) OptimizedSoftmax.Backward(p, dy, dx, scale, threads); else ScalarSoftmax.Backward(p, dy, dx, scale); }
    internal static void Fused(bool fast, FloatMatrix labels, FloatMatrix p, FloatMatrix dx, float coefficient = 1, float scale = 0, int threads = -1)
    { if (fast) OptimizedSoftmax.BackwardLogreg(labels, p, dx, coefficient, scale, threads); else ScalarSoftmax.BackwardLogreg(labels, p, dx, coefficient, scale); }
    internal static void ProbabilityBackward(bool fast, FloatMatrix labels, FloatMatrix p, FloatMatrix dx, float coefficient = 1, float scale = 0, int threads = -1)
    { if (fast) OptimizedLogregCost.Backward(labels, p, dx, coefficient, scale, threads); else ScalarLogregCost.Backward(labels, p, dx, coefficient, scale); }
    internal static LogregCostResult Cost(bool fast, FloatMatrix labels, FloatMatrix p, int k, int threads = -1) =>
        fast ? OptimizedLogregCost.Forward(labels, p, k, threads) : ScalarLogregCost.Forward(labels, p, k);

    private sealed class DropoutAdapter
    {
        private readonly ScalarDropout? _scalar;
        private readonly OptimizedDropout? _fast;
        internal DropoutAdapter(bool fast, float keep, bool enable, IDropoutMaskSource source, int threads = -1)
        { if (fast) _fast = new(keep, enable, source, threads); else _scalar = new(keep, enable, source); }
        internal void Forward(FloatMatrix x, FloatMatrix y, DropoutMode mode = DropoutMode.Train)
        { if (_fast != null) _fast.Forward(x, y, mode); else _scalar!.Forward(x, y, mode); }
        internal void Backward(FloatMatrix dy, FloatMatrix dx, float scale = 0)
        { if (_fast != null) _fast.Backward(dy, dx, scale); else _scalar!.Backward(dy, dx, scale); }
    }
    private static int Parity()
    {
        int[] casesList = [1, 2, 3, 5, 17, 128];
        for (int seed = 0; seed < 24; seed++)
        {
            var rng = new Random(100100 + seed); int cases = casesList[seed % casesList.Length], classes = 1 + seed % 17;
            var x = S(cases, classes); var dy = S(cases, classes); Fill(x, rng); Fill(dy, rng);
            if (seed % 5 == 0) { x[0, 0] = float.NaN; dy[cases - 1, classes - 1] = float.NegativeInfinity; }
            if (seed % 7 == 0) { x[cases - 1, classes - 1] = float.NegativeInfinity; dy[0, 0] = -0f; }
            var labels = C(1, cases); for (int n = 0; n < cases; n++) labels[0, n] = n % classes;
            var p = S(cases, classes); ScalarSoftmax.Forward(x, p); var beforeX = Values(x); var beforeP = Values(p); var beforeDy = Values(dy);
            foreach (int threads in new[] { 1, 2, 24, Environment.ProcessorCount, Environment.ProcessorCount })
            {
                var y = S(cases, classes); OptimizedSoftmax.Forward(x, y, threads); Bits(p, y, "softmax forward bits");
                var expected = S(cases, classes); var actual = S(cases, classes);
                float scale = new[] { 0f, 1f, .5f, -1f }[seed % 4]; expected.Fill(seed % 3 == 0 ? float.NaN : 2); actual.Fill(seed % 3 == 0 ? float.NaN : 2);
                ScalarSoftmax.Backward(p, dy, expected, scale); OptimizedSoftmax.Backward(p, dy, actual, scale, threads); Bits(expected, actual, "general backward bits");
                float coeff = new[] { 0f, 1f, -.75f, .3f }[seed % 4]; expected.Fill(3); actual.Fill(3);
                ScalarSoftmax.BackwardLogreg(labels, p, expected, coeff, scale); OptimizedSoftmax.BackwardLogreg(labels, p, actual, coeff, scale, threads); Bits(expected, actual, "fused bits");
                var pc = p.TransposeView(); expected = expected.TransposeView(); actual = actual.TransposeView(); expected.Fill(float.NaN); actual.Fill(float.NaN);
                ScalarLogregCost.Backward(labels, pc, expected, coeff, scale); OptimizedLogregCost.Backward(labels, pc, actual, coeff, scale, threads); Bits(expected, actual, "probability gradient bits");
                Bits(ScalarLogregCost.Forward(labels, pc, 5), OptimizedLogregCost.Forward(labels, pc, 5, threads));
                var mvS = new LogregMultiviewAccumulator(); var mvF = new LogregMultiviewAccumulator(true, threads);
                for (int v = 0; v < 10; v++)
                {
                    var view = C(classes, cases); Fill(view, rng);
                    if (seed % 7 == 0) view[0, 0] = float.NaN;
                    var phase = v == 0 ? MultiviewPhase.Start : v == 9 ? MultiviewPhase.End : MultiviewPhase.Continue;
                    var cs = mvS.Forward(labels, view, 5, phase); var cf = mvF.Forward(labels, view, 5, phase);
                    Bits(mvS.CopyProbabilities(), mvF.CopyProbabilities(), "multiview bits"); if (v == 9) Bits(cs!, cf!);
                }
                var dropX = x.TransposeView(); var dropDy = dy.TransposeView(); var sy = C(classes, cases); var fy = C(classes, cases);
                bool enable = seed % 4 != 0; float keep = new[] { 0f, 1f, .5f, .3f }[seed % 4];
                var sourceS = new SeededDropoutMaskSource((ulong)seed); var sourceF = new SeededDropoutMaskSource((ulong)seed);
                var ds = new ScalarDropout(keep, enable, sourceS); var df = new OptimizedDropout(keep, enable, sourceF, threads);
                foreach (var mode in new[] { DropoutMode.Train, DropoutMode.Evaluate, DropoutMode.Train })
                {
                    ds.Forward(dropX, sy, mode); df.Forward(dropX, fy, mode); Bits(sy, fy, "dropout forward bits");
                    var sdx = C(classes, cases); var fdx = C(classes, cases); sdx.Fill(float.NaN); fdx.Fill(float.NaN);
                    ds.Backward(dropDy, sdx, scale); df.Backward(dropDy, fdx, scale); Bits(sdx, fdx, "dropout backward bits");
                    TestAssert.Equal(sourceS.DrawOffset, sourceF.DrawOffset, "matching RNG cursor");
                }
            }
            Bits(beforeX, Values(x), "preserve logits"); Bits(beforeP, Values(p), "preserve probabilities"); Bits(beforeDy, Values(dy), "preserve dy");
        }
        // Actual batch-128 mask shape and explicit requested thread counts, including successive fills.
        var baseline = new SeededDropoutMaskSource(100024); var a = C(2048, 128); var b = C(2048, 128);
        baseline.Fill(a, .5f, 1);
        foreach (int threads in new[] { 1, 2, 24, Environment.ProcessorCount })
        {
            var source = new SeededDropoutMaskSource(100024); source.Fill(b, .5f, threads); Bits(a, b, "2048x128 RNG threads");
            var next = C(2048, 128); source.Fill(next, .5f, threads);
            TestAssert.True(!Values(a).SequenceEqual(Values(next)), "successive forward gets fresh mask");
            var resumed = new SeededDropoutMaskSource(100024, (ulong)a.ElementCount); resumed.Fill(b, .5f, threads); Bits(next, b, "resume mask from cursor");
        }
        float u = SeededDropoutMaskSource.Uniform(100025, 0); var one = C(1, 1);
        new SeededDropoutMaskSource(100025).Fill(one, u); TestAssert.Equal(0f, one[0, 0], "strict u<keep at equality");
        new SeededDropoutMaskSource(100025).Fill(one, MathF.BitIncrement(u)); TestAssert.Equal(1f, one[0, 0], "next representable keep");
        return 26;
    }
    private static int Numerical()
    {
        double worst = 0; int comparisons = 0;
        void Gradient(double numeric, double actual, string label)
        {
            worst = Math.Max(worst, Math.Abs(numeric - actual)); comparisons++;
            Near(numeric, actual, label, 3e-6);
        }
        foreach (int seed in new[] { 100201, 100202, 100203 })
        {
            var rng = new Random(seed); int cases = 2 + seed % 2, classes = 3 + seed % 3;
            var x = S(cases, classes); var dy = S(cases, classes); Fill(x, rng); Fill(dy, rng);
            var labels = C(1, cases); for (int n = 0; n < cases; n++) labels[0, n] = (n + 1) % classes;
            var z = new double[cases, classes]; for (int n = 0; n < cases; n++) for (int c = 0; c < classes; c++) z[n, c] = x[n, c];
            // Independent float64 objectives: normalizer from Math.Exp, never port functions.
            double Objective(bool loss)
            {
                double result = 0;
                for (int n = 0; n < cases; n++)
                {
                    double max = Enumerable.Range(0, classes).Max(c => z[n, c]);
                    var e = Enumerable.Range(0, classes).Select(c => Math.Exp(z[n, c] - max)).ToArray(); double total = e.Sum();
                    if (loss) result -= Math.Log(e[(int)labels[0, n]] / total);
                    else for (int c = 0; c < classes; c++) result += e[c] / total * dy[n, c];
                }
                return result;
            }
            foreach (bool fast in new[] { false, true })
            {
                var p = S(cases, classes); Forward(fast, x, p); var general = S(cases, classes); Backward(fast, p, dy, general);
                var fused = S(cases, classes); Fused(fast, labels, p, fused, .75f);
                const double epsilon = 1e-5;
                for (int n = 0; n < cases; n++) for (int c = 0; c < classes; c++)
                {
                    double old = z[n, c];
                    foreach (bool loss in new[] { false, true })
                    {
                        z[n, c] = old + epsilon; double plus = Objective(loss); z[n, c] = old - epsilon; double minus = Objective(loss); z[n, c] = old;
                        double numeric = (plus - minus) / (2 * epsilon);
                        Gradient(loss ? -.75 * numeric : numeric, loss ? fused[n, c] : general[n, c], "float64 logits gradient");
                    }
                }
                var mask = Enumerable.Range(0, classes * cases).Select(_ => rng.Next(2) == 0 ? 0f : 1f).ToArray();
                var dropout = new DropoutAdapter(fast, .5f, true, new FixedDropoutMaskSource(mask)); var dc = C(classes, cases);
                dropout.Forward(x.TransposeView(), dc); dropout.Backward(dy.TransposeView(), dc);
                double DropObjective()
                {
                    double result = 0;
                    for (int n = 0; n < cases; n++) for (int c = 0; c < classes; c++) result += z[n, c] * mask[c * cases + n] * dy[n, c];
                    return result;
                }
                for (int n = 0; n < cases; n++) for (int c = 0; c < classes; c++)
                {
                    double old = z[n, c]; z[n, c] += epsilon; double plus = DropObjective(); z[n, c] = old - epsilon; double minus = DropObjective(); z[n, c] = old;
                    Gradient((plus - minus) / (2 * epsilon), dc[c, n], "fixed-mask float64 dropout gradient");
                }
            }
        }
        Console.WriteLine($"ANX-10 NUMERICAL: {comparisons} central differences, epsilon=1e-5, tolerance=3e-6*(1+abs(numeric)), worst abs={worst:G9}.");
        return 6;
    }
    private static int Contracts()
    {
        var p = S(2, 3); p.Fill(.25f); var dy = S(2, 3); dy.Fill(1); var dx = S(2, 3); var labels = C(1, 2);
        foreach (bool fast in new[] { false, true })
        {
            Reject(() => Forward(fast, p.TransposeView(), dx)); Reject(() => Forward(fast, p, p));
            Reject(() => Forward(fast, p, S(3, 3))); Reject(() => Backward(fast, p, dy, p));
            Reject(() => Fused(fast, labels.TransposeView(), p, dx));
            labels[0, 1] = 3; dx.Fill(123); Reject(() => Fused(fast, labels, p, dx)); TestAssert.True(Values(dx).All(v => v == 123), "validate all labels before mutation");
            labels[0, 1] = float.NaN; Reject(() => Cost(fast, labels, p.TransposeView(), 5)); labels[0, 1] = 0;
            Reject(() => Cost(fast, labels, p.TransposeView(), 0)); Reject(() => ProbabilityBackward(fast, labels, p.TransposeView(), p.TransposeView()));
            // Original label float-to-int truncation (including a small negative value).
            labels[0, 0] = -.5f; labels[0, 1] = 1.9f; Fused(fast, labels, p, dx);
            Near(.75, dx[0, 0], "truncated label 0", 0); Near(.75, dx[1, 1], "truncated label 1", 0); labels.Fill(0);
            // Boolean scale contracts differ between hs and softmax/cost.
            var clean = S(2, 3); Backward(fast, p, dy, clean);
            dx.Fill(float.NaN); Backward(fast, p, dy, dx, .5f); Bits(clean, dx, "softmax scale .5 overwrites");
            Fused(fast, labels, p, clean); dx.Fill(float.NaN); Fused(fast, labels, p, dx, 1, -1); Bits(clean, dx, "fused scale -1 overwrites");
            var pc = p.TransposeView(); var cc = C(3, 2); ProbabilityBackward(fast, labels, pc, cc);
            var ac = C(3, 2); ac.Fill(float.NaN); ProbabilityBackward(fast, labels, pc, ac, 1, .5f); Bits(cc, ac, "probability scale .5 overwrites");
            var source = new CountingSource(); var dropout = new DropoutAdapter(fast, .5f, true, source);
            Reject(() => dropout.Backward(pc, ac)); Reject(() => dropout.Forward(p, ac));
            dropout.Forward(pc, ac); TestAssert.Equal(1, source.Calls, "one mask fill");
            var first = C(3, 2); dropout.Backward(pc, first); dropout.Backward(pc, ac); Bits(first, ac, "saved mask reuse");
            TestAssert.Equal(1, source.Calls, "backward never draws mask");
            ac.Fill(float.NaN); dropout.Backward(pc, ac, .5f); TestAssert.True(Values(ac).All(float.IsNaN), "hs nonzero adds");
            foreach (var mode in new[] { DropoutMode.Evaluate, DropoutMode.GradientCheck, DropoutMode.FeatureGeneration, DropoutMode.MultiviewTest })
            { dropout.Forward(pc, ac, mode); dropout.Backward(pc, ac); TestAssert.Equal(1, source.Calls, "nontraining no draws"); }
            var disabledSource = new CountingSource(); var disabled = new DropoutAdapter(fast, .5f, false, disabledSource);
            disabled.Forward(pc, ac); disabled.Backward(pc, ac); TestAssert.Equal(0, disabledSource.Calls, "disabled no draws");
            dropout.Forward(pc, ac); TestAssert.Equal(2, source.Calls, "new training forward redraws");
            Reject(() => dropout.Backward(C(2, 2), C(2, 2)));
            var inPlace = C(3, 2); inPlace.Fill(2); var e = new DropoutAdapter(fast, .5f, true, new FixedDropoutMaskSource([1, 0, 0, 1, 1, 0]));
            e.Forward(inPlace, inPlace); Bits(new float[] { 2, 0, 0, 2, 2, 0 }, Values(inPlace), "in-place dropout");
            e.Backward(inPlace, inPlace); Bits(new float[] { 2, 0, 0, 2, 2, 0 }, Values(inPlace), "in-place backward");
            var overlap = C(3, 4); Reject(() => e.Forward(overlap.Slice(0, 3, 0, 2), overlap.Slice(0, 3, 1, 2)));
            var a = new LogregMultiviewAccumulator(fast); Reject(() => a.Forward(labels, pc, 5, MultiviewPhase.End));
            a.Forward(labels, pc, 5, MultiviewPhase.Start); Reject(() => a.Forward(C(1, 3), C(3, 3), 5, MultiviewPhase.End));
            a.Forward(labels, pc, 5, MultiviewPhase.End); Reject(() => a.Forward(labels, pc, 5, MultiviewPhase.Continue));
        }
        Reject(() => OptimizedSoftmax.Forward(p, dx, 0)); Reject(() => OptimizedLogregCost.Forward(labels, p.TransposeView(), 1, 0));
        Reject(() => new OptimizedDropout(.5f, true, new CountingSource(), 0)); Reject(() => new FixedDropoutMaskSource([.25f]));
        // Strided native and canonical views include nonzero offsets and untouched sentinels.
        var xb = C(5, 7); var yb = C(5, 7); var db = C(5, 7); var gb = C(5, 7);
        xb.Fill(77); yb.Fill(77); db.Fill(77); gb.Fill(77);
        var xc = xb.Slice(1, 3, 2, 3); var yc = yb.Slice(1, 3, 2, 3); var dc = db.Slice(1, 3, 2, 3); var gc = gb.Slice(1, 3, 2, 3);
        var rng = new Random(100300); Fill(xc, rng); Fill(dc, rng); var compact = C(3, 3); compact.Add(xc, 0);
        var expected = S(3, 3); ScalarSoftmax.Forward(compact.TransposeView(), expected);
        OptimizedSoftmax.Forward(xc.TransposeView(), yc.TransposeView(), 2); Bits(expected.TransposeView(), yc, "strided forward");
        var sg = S(3, 3); ScalarSoftmax.Backward(expected, dc.TransposeView(), sg); OptimizedSoftmax.Backward(yc.TransposeView(), dc.TransposeView(), gc.TransposeView(), 0, 2);
        Bits(sg.TransposeView(), gc, "strided backward");
        var sl = C(1, 3); var labelBacking = C(1, 5); var vl = labelBacking.Slice(0, 1, 1, 3); vl.Fill(0);
        Bits(ScalarLogregCost.Forward(sl, expected.TransposeView(), 2), OptimizedLogregCost.Forward(vl, yc, 2, 2));
        ScalarSoftmax.BackwardLogreg(sl, expected, sg); OptimizedSoftmax.BackwardLogreg(vl, yc.TransposeView(), gc.TransposeView(), 1, 0, 2); Bits(sg.TransposeView(), gc, "strided fused");
        var dr = new ScalarDropout(.5f, true, new SeededDropoutMaskSource(101)); var df = new OptimizedDropout(.5f, true, new SeededDropoutMaskSource(101), 2);
        dr.Forward(compact, expected.TransposeView()); df.Forward(xc, yc); Bits(expected.TransposeView(), yc, "strided dropout");
        dr.Backward(dc, sg.TransposeView()); df.Backward(dc, gc); Bits(sg.TransposeView(), gc, "strided dropout backward");
        for (int f = 0; f < 5; f++) for (int n = 0; n < 7; n++) if (f < 1 || f >= 4 || n < 2 || n >= 5)
        { TestAssert.Equal(77f, yb[f, n], "forward sentinel"); TestAssert.Equal(77f, gb[f, n], "backward sentinel"); }
        // Every part of the exact fused condition, plus the independent cost skip condition.
        TestAssert.True(SoftmaxFusion.Select(true, 1, "cost.logreg", 0, 0), "sole local logreg fuses");
        TestAssert.True(!SoftmaxFusion.Select(false, 1, "cost.logreg", 0, 0), "flag disables fusion");
        TestAssert.True(!SoftmaxFusion.Select(true, 2, "cost.logreg", 0, 0), "multiple consumers general");
        TestAssert.True(!SoftmaxFusion.Select(true, 1, "cost.other", 0, 0), "other consumer general");
        TestAssert.True(!SoftmaxFusion.Select(true, 1, "cost.logreg", 0, 1), "remote consumer general");
        TestAssert.True(!SoftmaxFusion.CostComputesProbabilityGradient(1, "softmax", 0, 0), "cost skips fused scratch");
        TestAssert.True(SoftmaxFusion.CostComputesProbabilityGradient(2, "softmax", 0, 0), "cost fanout work");
        TestAssert.True(SoftmaxFusion.CostComputesProbabilityGradient(1, "fc", 0, 0), "cost producer type work");
        TestAssert.True(SoftmaxFusion.CostComputesProbabilityGradient(1, "softmax", 1, 0), "cost remote work");
        return 5;
    }
    private sealed class CountingSource : IDropoutMaskSource
    {
        internal int Calls;
        public void Fill(FloatMatrix mask, float keep, int maxDegreeOfParallelism = 1) { Calls++; mask.Fill(1); }
    }
    private static int Real(string original, bool full)
    {
        var net = ShapeTests.Parse145(original); var shapes = ShapeInference.Infer(net); int groups = 0;
        foreach (string name in new[] { "hs1a", "hs1b", "hs2a", "hs2b" })
        {
            var layer = net[name]; var shape = shapes[name]; var p = (DropoutLayerParams)layer.TypedParams;
            TestAssert.Equal(2048, shape.Outputs, name + " width"); TestAssert.Equal(.5, p.Keep, name + " keep"); TestAssert.True(p.Enable, name + " enabled");
            TestAssert.True(net[layer.Inputs[0]].Type == "neuron", "dropout follows detached neuron");
            int cases = 128; var x = C(2048, cases); var dy = C(2048, cases); Fill(x, new Random(100401)); Fill(dy, new Random(100402));
            var sy = C(2048, cases); var sg = C(2048, cases); var scalar = new ScalarDropout((float)p.Keep, p.Enable, new SeededDropoutMaskSource(100403));
            scalar.Forward(x, sy); scalar.Backward(dy, sg);
            foreach (int threads in new[] { 1, 2, Environment.ProcessorCount })
            {
                var fast = new OptimizedDropout((float)p.Keep, p.Enable, new SeededDropoutMaskSource(100403), threads);
                var fy = C(2048, cases); var fg = C(2048, cases); fast.Forward(x, fy); fast.Backward(dy, fg); Bits(sy, fy, name + " batch128 forward"); Bits(sg, fg, name + " batch128 backward");
                fg.Fill(float.NaN); fast.Backward(dy, fg); Bits(sg, fg, "overwrite old gradient");
            }
            scalar.Forward(x, sy, DropoutMode.Evaluate);
            var eval = new OptimizedDropout(.5f, true, new CountingSource()); var ey = C(2048, cases); eval.Forward(x, ey, DropoutMode.Evaluate); Bits(sy, ey, "batch128 eval"); groups++;
        }
        var lp = (LogregCostLayerParams)net["logprob"].TypedParams;
        TestAssert.Equal(1000, shapes["probs"].Outputs, "1000 classes"); TestAssert.Equal(5, lp.TopK, "top5"); TestAssert.Equal(1d, lp.Coeff, "coeff");
        TestAssert.SequenceEqual(new[] { "labels", "probs" }, net["logprob"].Inputs, "cost input order");
        TestAssert.True(SoftmaxFusion.Select(net, "probs"), "145 selects fused path"); TestAssert.True(!SoftmaxFusion.Select(net, "probs", false), "145 opt out");
        var logits = S(128, 1000); Fill(logits, new Random(100410)); var labels = C(1, 128); labels[0, 127] = 999;
        var probabilities = S(128, 1000); ScalarSoftmax.Forward(logits, probabilities); var grad = S(128, 1000); ScalarSoftmax.BackwardLogreg(labels, probabilities, grad);
        var cost = ScalarLogregCost.Forward(labels, probabilities.TransposeView(), 5);
        var route = SoftmaxBackwardRoute.FromNetwork(net, "probs");
        TestAssert.True(route.FuseLogreg && route.Coefficient == 1, "145 parsed dispatch descriptor");
        var selected = S(128, 1000);
        ScalarSoftmax.BackwardSelected(route, probabilities, selected, labels: labels); Bits(grad, selected, "scalar selected fused route");
        OptimizedSoftmax.BackwardSelected(route, probabilities, selected, labels: labels); Bits(grad, selected, "optimized selected fused route");
        Reject(() => ScalarSoftmax.BackwardSelected(route, probabilities, selected));
        // General route selected from the SAME 145 graph with its explicit flag off.
        // Use tiny matrices here; full-size general parity follows in the opt-in group.
        var generalRoute = SoftmaxBackwardRoute.FromNetwork(net, "probs", false);
        var tiny = S(2, 3); tiny.Fill(.25f); var tinyDy = S(2, 3); tinyDy.Fill(2);
        var tinyExpected = S(2, 3); var tinyActual = S(2, 3); ScalarSoftmax.Backward(tiny, tinyDy, tinyExpected);
        ScalarSoftmax.BackwardSelected(generalRoute, tiny, tinyActual, upstream: tinyDy); Bits(tinyExpected, tinyActual, "scalar selected general route");
        OptimizedSoftmax.BackwardSelected(generalRoute, tiny, tinyActual, upstream: tinyDy); Bits(tinyExpected, tinyActual, "optimized selected general route");
        Reject(() => OptimizedSoftmax.BackwardSelected(generalRoute, tiny, tinyActual));
        foreach (int threads in new[] { 1, 2, 24, Environment.ProcessorCount })
        {
            var p = S(128, 1000); var dx = S(128, 1000); OptimizedSoftmax.Forward(logits, p, threads); Bits(probabilities, p, "batch128 probs");
            OptimizedSoftmax.BackwardLogreg(labels, p, dx, 1, 0, threads); Bits(grad, dx, "batch128 fused");
            Bits(cost, OptimizedLogregCost.Forward(labels, p.TransposeView(), 5, threads));
            // Single fused case repeated in a batch must not be divided by N.
            Near(1 - p[0, 0], dx[0, 0], "no batch scaling", 0);
        }
        groups++;
        // General softmax at full 1000-way geometry is intentionally opt-in (O(C²*N)).
        if (full)
        {
            var dy = S(128, 1000); Fill(dy, new Random(100411)); var expected = S(128, 1000); ScalarSoftmax.Backward(probabilities, dy, expected);
            foreach (int threads in new[] { 1, 2, Environment.ProcessorCount })
            { var actual = S(128, 1000); OptimizedSoftmax.Backward(probabilities, dy, actual, 0, threads); Bits(expected, actual, "full batch128 general backward"); }
            groups++;
        }
        groups += Fanout(net, shapes, full ? 128 : 3);
        return groups;
    }
    private static int Fanout(NetworkDefinition net, NetworkShapes shapes, int cases)
    {
        TestAssert.SequenceEqual(new[] { "hs1a", "hs1b" }, net["fc2048ba"].Inputs, "FC pair a order");
        TestAssert.SequenceEqual(new[] { "hs1b", "hs1a" }, net["fc2048bb"].Inputs, "FC pair b order");
        var rng = new Random(100420); var ba = OptimizedFullyConnected.FromLayer(net["fc2048ba"], shapes["fc2048ba"]);
        var bb = OptimizedFullyConnected.FromLayer(net["fc2048bb"], shapes["fc2048bb"]);
        var dyA = S(cases, 2048); var dyB = S(cases, 2048); Fill(dyA, rng); Fill(dyB, rng);
        var weightsA = new[] { S(2048, 2048), S(2048, 2048) }; var weightsB = new[] { S(2048, 2048), S(2048, 2048) };
        foreach (var w in weightsA.Concat(weightsB)) Fill(w, rng);
        foreach (string producer in new[] { "hs1a", "hs1b" })
        {
            var consumers = net.Layers.Where(l => l.Inputs.Contains(producer)).ToArray(); TestAssert.Equal(2, consumers.Length, "two FC consumers");
            TestAssert.SequenceEqual(new[] { 0, 1 }, consumers.Select(l => l.TypedParams.Gpu).Order().ToArray(), "one per GPU");
            int ai = producer == "hs1a" ? 0 : 1, bi = 1 - ai;
            var a = S(cases, 2048); var b = S(cases, 2048); a.Fill(float.NaN); b.Fill(float.NaN);
            ba.BackwardInput(ai, dyA, weightsA[ai], a, 0); bb.BackwardInput(bi, dyB, weightsB[bi], b, 0);
            // Check sampled real FC contributions against independent ascending scalar dots.
            foreach (int n in new[] { 0, cases - 1 }) foreach (int f in new[] { 0, 1023, 2047 })
            {
                float ea = 0, eb = 0;
                for (int o = 0; o < 2048; o++) { ea += dyA[n, o] * weightsA[ai][f, o]; eb += dyB[n, o] * weightsB[bi][f, o]; }
                Near(ea, a[n, f], "FC local contribution", 0); Near(eb, b[n, f], "FC remote contribution", 0);
            }
            var merged = C(2048, cases); var mask = Enumerable.Range(0, 2048 * cases).Select(i => i % 3 == 0 ? 0f : 1f).ToArray();
            var x = C(2048, cases); x.Fill(1); var saved = C(2048, cases); var dropout = new OptimizedDropout(.5f, true, new FixedDropoutMaskSource(mask)); dropout.Forward(x, saved);
            foreach (bool reverse in new[] { false, true })
            {
                // hs1a local=a, hs1b local=b; arrival order cannot change which is local.
                FloatMatrix local = producer == "hs1a" ? a : b, remote = producer == "hs1a" ? b : a;
                var first = reverse ? remote : local; var second = reverse ? local : remote;
                var localBuffer = C(2048, cases); var remoteBuffer = C(2048, cases); localBuffer.Fill(float.NaN); remoteBuffer.Fill(float.NaN);
                (reverse ? remoteBuffer : localBuffer).Add(first.TransposeView(), 0);
                (reverse ? localBuffer : remoteBuffer).Add(second.TransposeView(), 0);
                merged.Fill(float.NaN); merged.Add(localBuffer, 0); merged.Add(remoteBuffer, 1);
                var result = C(2048, cases); result.Fill(float.NaN); dropout.Backward(merged, result);
                for (int f = 0; f < 2048; f++) for (int n = 0; n < cases; n++)
                {
                    float expected = (local[n, f] + remote[n, f]) * mask[f * cases + n];
                    if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(result[f, n])) throw new TestFailureException("FC fan-out merge then saved dropout mask");
                }
            }
        }
        return 2;
    }
}
