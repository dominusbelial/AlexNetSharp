using System.Globalization;
using System.Numerics;
using System.Text.Json.Nodes;
using AlexNetSharp.Core.Layers.Norm;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class NormTests
{
    public static int RunAll(string repo, string original, bool full)
    {
        int fixtures = Fixtures(repo), parity = Parity(), numeric = Numerical(), contracts = Contracts();
        int real = RealGeometry(original, full);
        Console.WriteLine($"NORM TESTS PASSED: {fixtures} fixture, {parity} parity, {numeric} numerical, {contracts} contract and {real} real-geometry groups.");
        return fixtures + parity + numeric + contracts + real;
    }

    internal sealed class Engine
    {
        private readonly ScalarNorm? _scalar;
        private readonly OptimizedNorm? _fast;
        internal Engine(bool fast, NormGeometry geometry, string kind, float scale, float power = 0.75f,
            float minimum = 2, bool blocked = false, int threads = -1)
        {
            if (fast) _fast = new("norm-test", geometry, kind, scale, power, minimum, blocked, threads);
            else _scalar = new("norm-test", geometry, kind, scale, power, minimum, blocked);
        }
        internal void Forward(FloatMatrix x, FloatMatrix d, FloatMatrix y)
        { if (_fast != null) _fast.Forward(x, d, y); else _scalar!.Forward(x, d, y); }
        internal void Backward(FloatMatrix x, FloatMatrix dy, FloatMatrix d, FloatMatrix y, FloatMatrix dx, float scale = 0)
        { if (_fast != null) _fast.Backward(x, dy, d, y, dx, scale); else _scalar!.Backward(x, dy, d, y, dx, scale); }
    }

    private static int Fixtures(string repo)
    {
        var cases = JsonNode.Parse(File.ReadAllText(Path.Combine(repo, "tests/fixtures/norm-cases.json")))!["fixtures"]!.AsArray();
        foreach (var f in cases.Select(n => n!))
        {
            var geometry = new NormGeometry(I(f, "width"), I(f, "channels"), I(f, "size"));
            int n = I(f, "cases"); string kind = f["kind"]!.GetValue<string>();
            var x = Read(geometry.Rows, n, f["input"]!); var dy = Read(geometry.Rows, n, f["upstream"]!);
            var savedX = Copy(x); var savedDy = Copy(dy);
            foreach (bool fast in new[] { false, true })
            {
                var engine = new Engine(fast, geometry, kind, F(f["scale"]!), F(f["power"]!), F(f["minimum"]!), f["blocked"]!.GetValue<bool>());
                var d = new FloatMatrix(x.Rows, n); var y = new FloatMatrix(x.Rows, n); var dx = new FloatMatrix(x.Rows, n);
                d.Fill(float.NaN); y.Fill(float.NaN); dx.Fill(float.NaN);
                engine.Forward(x, d, y);
                Check(f["denominators"]!, d, "fixture denominators"); Check(f["output"]!, y, "fixture output", signedZero: true);
                var savedD = Copy(d); var savedY = Copy(y);
                engine.Backward(x, dy, d, y, dx); Check(f["backward"]!, dx, "fixture backward");
                dx.Fill(3); engine.Backward(x, dy, d, y, dx, 1); Check(f["backward"]!, dx, "fixture addition", 3);
                dx.Fill(float.NaN); engine.Backward(x, dy, d, y, dx, 1);
                Each(dx, v => TestAssert.True(float.IsNaN(v), "NaN target retained"));
                Bits(savedX, x, "input preserved"); Bits(savedDy, dy, "upstream preserved");
                Bits(savedD, d, "cache preserved"); Bits(savedY, y, "saved output preserved");
            }
        }
        return cases.Count * 2;
    }

    private static int Parity()
    {
        for (int seed = 0; seed < 36; seed++)
        {
            var rng = new Random(81000 + seed);
            string kind = seed % 2 == 0 ? "cmrnorm" : "rnorm";
            var g = new NormGeometry(rng.Next(1, 8), rng.Next(1, 10), rng.Next(1, 8));
            int cases = new[] { 1, 2, Vector<float>.Count - 1, Vector<float>.Count, Vector<float>.Count + 1, 17, 128 }[seed % 7];
            float scale = seed % 3 == 0 ? 0.03f : 0.00002f, minimum = seed % 4 == 0 ? 1 : 2;
            float power = new[] { 0f, 0.5f, 0.75f, 1f, 1.25f }[seed % 5];
            bool blocked = seed % 3 == 0;
            var x = new FloatMatrix(g.Rows, cases); var dy = new FloatMatrix(g.Rows, cases);
            Fill(x, rng); Fill(dy, rng);
            if (seed % 5 == 0)
            {
                float[] specials = [-0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1e30f, -1e30f];
                for (int i = 0; i < Math.Min(x.ElementCount, specials.Length * 5); i++) x[i / cases, i % cases] = specials[i % specials.Length];
                if (dy.ElementCount > 9) dy[0, 0] = float.NaN;
            }
            var scalar = new Engine(false, g, kind, scale, power, minimum: minimum, blocked: blocked);
            var sy = new FloatMatrix(g.Rows, cases); var sd = new FloatMatrix(g.Rows, cases); var sx = new FloatMatrix(g.Rows, cases);
            scalar.Forward(x, sd, sy);
            foreach (float accumulate in new[] { 0f, 1f, 0.5f, -1f })
            {
                sx.Fill(accumulate == 0 ? float.NaN : 3); scalar.Backward(x, dy, sd, sy, sx, accumulate);
                foreach (int threads in new[] { 1, 2, Environment.ProcessorCount, Environment.ProcessorCount })
                {
                    var fast = new Engine(true, g, kind, scale, power, minimum: minimum, blocked: blocked, threads: threads);
                    var y = new FloatMatrix(g.Rows, cases); var d = new FloatMatrix(g.Rows, cases); var dx = new FloatMatrix(g.Rows, cases);
                    y.Fill(float.NaN); d.Fill(float.NaN); dx.Fill(accumulate == 0 ? float.NaN : 3);
                    fast.Forward(x, d, y); fast.Backward(x, dy, d, y, dx, accumulate);
                    Bits(sd, d, "random cache"); Bits(sy, y, "random forward"); Bits(sx, dx, "random backward");
                }
            }
        }
        return 36;
    }

    private static int Numerical()
    {
        int checks = 0; double worst = 0, worstDouble = 0;
        foreach (var spec in new[] { ("cmrnorm", 7, 2, 5, false), ("rnorm", 2, 6, 5, false),
                                    ("cmrnorm", 7, 1, 4, false), ("rnorm", 1, 6, 4, false),
                                    ("cmrnorm", 7, 1, 3, true) })
        {
            var (kind, channels, width, size, blocked) = spec;
            int cases = 2, rows = channels * width * width;
            var rng = new Random(82000 + checks); var x = new FloatMatrix(rows, cases); var dy = new FloatMatrix(rows, cases);
            Fill(x, rng); Fill(dy, rng);
            double[] values = Flatten(x), upstream = Flatten(dy);
            const float scale = 0.125f; const double power = 0.75, minimum = 2;
            // Independent output-centric membership/Jacobian, all coordinates including edges.
            var members = new int[rows][];
            for (int r = 0; r < rows; r++)
            {
                int c = r / (width * width), y = r / width % width, z = r % width;
                int start = blocked ? c / size * size : c - size / 2;
                members[r] = Enumerable.Range(0, rows).Where(j => kind == "cmrnorm"
                    ? j / width % width == y && j % width == z && j / (width * width) >= start && j / (width * width) < start + size
                    : j / (width * width) == c && j / width % width >= y - size / 2 && j / width % width < y - size / 2 + size &&
                        j % width >= z - size / 2 && j % width < z - size / 2 + size).ToArray();
            }
            var analytic = new double[values.Length];
            for (int r = 0; r < rows; r++) for (int n = 0; n < cases; n++)
            {
                int i = r * cases + n;
                double d = (kind == "cmrnorm" ? minimum : 1) + scale * members[r].Sum(j => values[j * cases + n] * values[j * cases + n]);
                double a = values[i] * Math.Pow(d, -power);
                analytic[i] += upstream[i] * Math.Pow(d, -power);
                foreach (int j in members[r]) analytic[j * cases + n] += -2 * scale * power * a * upstream[i] / d * values[j * cases + n];
            }
            var numerical = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                double v = values[i]; values[i] = v + 1e-5; double plus = Loss();
                values[i] = v - 1e-5; double minus = Loss(); values[i] = v;
                numerical[i] = (plus - minus) / 2e-5;
                double error = Math.Abs(analytic[i] - numerical[i]); worstDouble = Math.Max(worstDouble, error);
                TestAssert.True(error <= 2e-8 * (1 + Math.Abs(numerical[i])), "float64 norm derivative");
            }
            foreach (bool fast in new[] { false, true })
            {
                var engine = new Engine(fast, new(width, channels, size), kind, scale, minimum: (float)minimum, blocked: blocked);
                var d = new FloatMatrix(rows, cases); var y = new FloatMatrix(rows, cases); var dx = new FloatMatrix(rows, cases);
                engine.Forward(x, d, y); engine.Backward(x, dy, d, y, dx);
                for (int i = 0; i < numerical.Length; i++)
                {
                    double error = Math.Abs(dx[i / cases, i % cases] - numerical[i]); worst = Math.Max(worst, error);
                    TestAssert.True(error <= 4e-6 * (1 + Math.Abs(numerical[i])), "float32 numerical norm derivative"); checks++;
                }
            }
            double Loss()
            {
                double loss = 0;
                for (int r = 0; r < rows; r++) for (int n = 0; n < cases; n++)
                {
                    double d = kind == "cmrnorm" ? minimum : 1;
                    double sum = 0; foreach (int j in members[r]) sum += values[j * cases + n] * values[j * cases + n];
                    d += scale * sum; loss += upstream[r * cases + n] * values[r * cases + n] * Math.Pow(d, -power);
                }
                return loss;
            }
        }
        Console.WriteLine($"NORM NUMERICAL: {checks} float32 comparisons, worst abs {worst:G9}; independent float64 worst {worstDouble:G9}.");
        return 5 * 2;
    }

    private static int Contracts()
    {
        foreach (bool fast in new[] { false, true })
        {
            var engine = new Engine(fast, new(2, 3, 5), "cmrnorm", 0.1f);
            FloatMatrix Parent() { var m = new FloatMatrix(14, 20); m.Fill(1234); return m; }
            var xp = Parent(); var dp = Parent(); var yp = Parent(); var gp = Parent(); var tp = Parent();
            var x = xp.Slice(1, 12, 2, 17); var d = dp.Slice(1, 12, 2, 17); var y = yp.Slice(1, 12, 2, 17);
            var dy = gp.Slice(1, 12, 2, 17); var dx = tp.Slice(1, 12, 2, 17);
            Fill(x, new Random(83000)); Fill(dy, new Random(83001));
            engine.Forward(x, d, y); engine.Backward(x, dy, d, y, dx);
            var sy = new FloatMatrix(12, 17); var sd = new FloatMatrix(12, 17); var sx = new FloatMatrix(12, 17);
            var scalar = new Engine(false, new(2, 3, 5), "cmrnorm", 0.1f);
            scalar.Forward(Copy(x), sd, sy); scalar.Backward(Copy(x), Copy(dy), sd, sy, sx);
            Bits(sd, d, "strided cache"); Bits(sy, y, "strided output"); Bits(sx, dx, "strided backward");
            foreach (var p in new[] { xp, dp, yp, gp, tp })
                for (int r = 0; r < p.Rows; r++) for (int n = 0; n < p.Cols; n++)
                    if (r == 0 || r == 13 || n < 2 || n == 19) Same(1234, p[r, n], "sentinel");
            // Independently inject caches: with x=0 the cross term vanishes; an
            // all-16 denominator must control the direct term instead of recomputation.
            x.Fill(0); d.Fill(16); y.Fill(0); dy.Fill(1); dx.Fill(float.NaN);
            engine.Backward(x, dy, d, y, dx); Each(dx, v => Same(0.125f, v, "uses supplied denominator"));
            // Changed saved acts must control the preliminaries, even when they are
            // inconsistent with x. C=3<size=5 means every window includes all channels.
            x.Fill(1); d.Fill(1); y.Fill(0); engine.Backward(x, dy, d, y, dx); Each(dx, v => Same(1, v, "zero saved acts"));
            y.Fill(1); engine.Backward(x, dy, d, y, dx); Each(dx, v => Near(0.55, v, "uses supplied output"));
            dx.Fill(42); var before = Copy(dx);
            Throws(() => engine.Forward(x, d, x), "norm-test", "storage");
            Throws(() => engine.Forward(x, x, y), "storage");
            Throws(() => engine.Forward(x, d, d), "storage");
            Throws(() => engine.Backward(x, dy, d, y, dy), "storage");
            Throws(() => engine.Forward(x.TransposeView(), d, y), "C-order");
            Throws(() => engine.Backward(x, new FloatMatrix(11, 17), d, y, dx), "C-order");
            Throws(() => engine.Forward(x, new FloatMatrix(11, 17), y), "C-order");
            Bits(before, dx, "failed validation writes nothing");
            // IEEE overflow is intentional: float32 square overflows and inf*0 is NaN.
            var overflow = new Engine(fast, new(1, 1, 5), "cmrnorm", 0.1f);
            var ox = new FloatMatrix(1, 17); var od = new FloatMatrix(1, 17); var oy = new FloatMatrix(1, 17); var og = new FloatMatrix(1, 17); var ot = new FloatMatrix(1, 17);
            ox.Fill(1e30f); og.Fill(1); overflow.Forward(ox, od, oy);
            Each(od, v => TestAssert.True(float.IsPositiveInfinity(v), "square overflow denominator")); Each(oy, v => Same(0, v, "overflow finite input output"));
            ox.Fill(float.PositiveInfinity); overflow.Forward(ox, od, oy); Each(oy, v => TestAssert.True(float.IsNaN(v), "inf times zero"));
            overflow.Backward(ox, og, od, oy, ot); Each(ot, v => TestAssert.True(float.IsNaN(v), "inf backward"));
        }
        Throws(() => new ScalarNorm("unsupported-name", new(1, 1, 5), "bad", 1, 1), "unsupported-name", "unsupported in this port");
        Throws(() => new OptimizedNorm("bad", new(0, 1, 5), "rnorm", 1, 1), "geometry");
        Throws(() => new OptimizedNorm("bad", new(1, 1, 5), "rnorm", 1, 1, maxDegreeOfParallelism: 0), "maxDegree");
        return 5;
    }

    private static int RealGeometry(string original, bool full)
    {
        var network = ShapeTests.Parse145(original); var shapes = ShapeInference.Infer(network); int count = 0;
        foreach (string tower in new[] { "a", "b" })
        {
            TestAssert.Equal("conv1" + tower, network["rnorm1" + tower].Inputs.Single(), "signed conv1 -> norm");
            TestAssert.True(network["conv1" + tower].DetachedNeuronLayer == null, "conv1 pre-ReLU");
            TestAssert.Equal("rnorm1" + tower, network["pool1" + tower].Inputs.Single(), "norm -> pool1");
            TestAssert.Equal("pool1" + tower, network["pool1" + tower + "_neuron"].Inputs.Single(), "pool1 -> relu");
            TestAssert.Equal("conv2" + tower, network["conv2" + tower + "_neuron"].Inputs.Single(), "conv2 -> relu");
            TestAssert.Equal("conv2" + tower + "_neuron", network["rnorm2" + tower].Inputs.Single(), "relu -> cmrnorm");
            TestAssert.Equal("rnorm2" + tower, network["cnorm2" + tower].Inputs.Single(), "cmrnorm -> spatial");
            TestAssert.Equal("cnorm2" + tower, network["pool2" + tower].Inputs.Single(), "spatial -> pool2");
        }
        foreach (var layer in network.Layers.Where(l => l.Type is "cmrnorm" or "rnorm"))
        {
            var s = shapes[layer.Name]; var scalar = ScalarNorm.FromLayer(layer, s); var fast = OptimizedNorm.FromLayer(layer, s);
            var g = scalar.Geometry;
            TestAssert.Equal(layer.Name.StartsWith("rnorm1", StringComparison.Ordinal) ? 55 : 27, g.ImageSize, "145 width");
            TestAssert.Equal(g.ImageSize == 55 ? 48 : 128, g.Channels, "145 channels"); TestAssert.Equal(5, g.Size, "145 window");
            Same(layer.Type == "cmrnorm" ? 2e-5f : 4e-5f, scalar.AddScale, "parser scale division");
            Same(0.75f, scalar.Power, "145 power"); Same(layer.Type == "cmrnorm" ? 2 : 1, scalar.MinDiv, "145 minimum");
            TestAssert.True(!scalar.Blocked, "145 unblocked"); TestAssert.Equal(g.Rows, s.Outputs, "shape");
            int cases = full ? 17 : 3;
            var x = new FloatMatrix(g.Rows, cases); var dy = new FloatMatrix(g.Rows, cases);
            Fill(x, new Random(84145 + count)); Fill(dy, new Random(85145 + count));
            // conv2's input to cmrnorm, and its following spatial norm, are nonnegative.
            if (g.ImageSize == 27) for (int r = 0; r < x.Rows; r++) for (int n = 0; n < cases; n++) x[r, n] = Math.Abs(x[r, n]);
            var sd = new FloatMatrix(g.Rows, cases); var sy = new FloatMatrix(g.Rows, cases); var sx = new FloatMatrix(g.Rows, cases);
            scalar.Forward(x, sd, sy); scalar.Backward(x, dy, sd, sy, sx);
            foreach (int threads in full ? new[] { 1, 2, Environment.ProcessorCount, Environment.ProcessorCount } : new[] { Environment.ProcessorCount })
            {
                fast = OptimizedNorm.FromLayer(layer, s, threads);
                var d = new FloatMatrix(g.Rows, cases); var y = new FloatMatrix(g.Rows, cases); var dx = new FloatMatrix(g.Rows, cases);
                fast.Forward(x, d, y); fast.Backward(x, dy, d, y, dx);
                Bits(sd, d, "145 denominator"); Bits(sy, y, "145 output"); Bits(sx, dx, "145 backward");
                Each(dx, v => TestAssert.True(float.IsFinite(v), "145 finite"));
                if (full)
                {
                    scalar.Backward(x, dy, sd, sy, sx, 1); fast.Backward(x, dy, d, y, dx, 1); Bits(sx, dx, "145 accumulate");
                    scalar.Backward(x, dy, sd, sy, sx); // restore the overwrite oracle
                }
            }
            count++;
        }
        Console.WriteLine($"NORM REAL GEOMETRY: all six 145 layers, N={(full ? 17 : 3)}; exact configured parameters, forward/cache/backward bit parity and both tower chain orders.");
        return count + 1;
    }

    internal static void Fill(FloatMatrix m, Random rng)
    { for (int r = 0; r < m.Rows; r++) for (int n = 0; n < m.Cols; n++) m[r, n] = rng.Next(-16, 17) / 4f; }
    private static FloatMatrix Copy(FloatMatrix m)
    { var result = new FloatMatrix(m.Rows, m.Cols); for (int r = 0; r < m.Rows; r++) for (int n = 0; n < m.Cols; n++) result[r, n] = m[r, n]; return result; }
    private static double[] Flatten(FloatMatrix m) => Enumerable.Range(0, m.ElementCount).Select(i => (double)m[i / m.Cols, i % m.Cols]).ToArray();
    private static int I(JsonNode n, string key) => n[key]!.GetValue<int>();
    private static float F(JsonNode n) => n.ToJsonString().StartsWith('"') ? n.GetValue<string>() switch
    { "NaN" => float.NaN, "Infinity" => float.PositiveInfinity, "-Infinity" => float.NegativeInfinity, "-0" => -0f, _ => throw new Exception("Bad fixture") }
        : float.Parse(n.ToJsonString(), CultureInfo.InvariantCulture);
    private static FloatMatrix Read(int rows, int cols, JsonNode data)
    { var m = new FloatMatrix(rows, cols); for (int i = 0; i < m.ElementCount; i++) m[i / cols, i % cols] = F(data[i]!); return m; }
    private static void Check(JsonNode data, FloatMatrix m, string label, float add = 0, bool signedZero = false)
    {
        for (int i = 0; i < m.ElementCount; i++)
        {
            double v = data[i]!.ToJsonString().StartsWith('"') ? F(data[i]!) : double.Parse(data[i]!.ToJsonString(), CultureInfo.InvariantCulture);
            double expected = add == 0 ? v : v + add; float actual = m[i / m.Cols, i % m.Cols];
            if (double.IsNaN(expected)) TestAssert.True(float.IsNaN(actual), label + " NaN");
            else if (double.IsInfinity(expected) || signedZero && expected == 0) Same((float)expected, actual, label);
            else Near(expected, actual, label);
        }
    }
    private static void Near(double expected, float actual, string label)
    { TestAssert.True(float.IsFinite(actual) && Math.Abs(expected - actual) <= 4e-6 * (1 + Math.Abs(expected)), $"{label}: {expected:G9} vs {actual:G9}"); }
    private static void Same(float expected, float actual, string label) => TestAssert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual), label);
    private static void Bits(FloatMatrix expected, FloatMatrix actual, string label)
    { for (int r = 0; r < expected.Rows; r++) for (int n = 0; n < expected.Cols; n++) Same(expected[r, n], actual[r, n], label); }
    private static void Each(FloatMatrix m, Action<float> f)
    { for (int r = 0; r < m.Rows; r++) for (int n = 0; n < m.Cols; n++) f(m[r, n]); }
    private static void Throws(Action f, params string[] messages)
    {
        try { f(); }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        { foreach (string m in messages) TestAssert.True(e.Message.Contains(m, StringComparison.Ordinal), "error " + m); return; }
        throw new TestFailureException("Expected rejection");
    }
}
