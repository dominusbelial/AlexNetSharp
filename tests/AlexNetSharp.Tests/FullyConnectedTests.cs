using System.Globalization;
using System.Numerics;
using System.Text.Json.Nodes;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Layers.FullyConnected;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class FullyConnectedTests
{
    internal static FloatMatrix M(int rows, int cols) => new(rows, cols, MatrixLayout.ColumnMajor);
    internal static void Fill(FloatMatrix m, Random rng)
    {
        for (int c = 0; c < m.Cols; c++) for (int r = 0; r < m.Rows; r++) m[r, c] = (float)(rng.NextDouble() - .5);
    }
    internal static float[] Values(FloatMatrix m) => Enumerable.Range(0, m.Cols).SelectMany(c => Enumerable.Range(0, m.Rows).Select(r => m[r, c])).ToArray();
    internal static void Bits(float[] expected, FloatMatrix actual, string label)
    {
        var a = Values(actual); TestAssert.Equal(expected.Length, a.Length, label + " count");
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(a[i]))
                throw new TestFailureException($"{label}[{i}]: bits {expected[i]:R} != {a[i]:R}");
    }
    private static float F(JsonNode n) => n is JsonValue v && v.TryGetValue<string>(out var s)
        ? s == "-0" ? -0f : float.Parse(s, CultureInfo.InvariantCulture) : n.GetValue<float>();
    private static FloatMatrix Read(int rows, int cols, JsonNode node)
    {
        var m = M(rows, cols); var a = node.AsArray();
        TestAssert.Equal(rows * cols, a.Count, "fixture length");
        for (int i = 0; i < a.Count; i++) m[i % rows, i / rows] = F(a[i]!);
        return m;
    }
    private static void Check(JsonNode node, FloatMatrix m, string label, float offset = 0)
    {
        var actual = Values(m); var expected = node.AsArray();
        TestAssert.Equal(expected.Count, actual.Length, label);
        for (int i = 0; i < actual.Length; i++)
        {
            float e = F(expected[i]!) + offset, a = actual[i];
            TestAssert.True(float.IsNaN(e) ? float.IsNaN(a) : e == a, $"{label}[{i}] expected {e:R}, got {a:R}");
        }
    }
    private static void Reject(Action action, string text)
    {
        try { action(); } catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        { TestAssert.True(ex.Message.Contains(text, StringComparison.Ordinal), "rejection: " + ex.Message); return; }
        throw new TestFailureException("Missing rejection: " + text);
    }
    public static int RunAll(string repo, string original, bool full)
    {
        int fixture = Fixtures(repo), parity = Parity(), numeric = Numerical(), contracts = Contracts(original);
        int real = Real(original, full), fanout = Fanout(repo, original);
        Console.WriteLine($"FC TESTS PASSED: {fixture} fixture, {parity} parity, {numeric} numerical, {contracts} contract, {real} real-geometry, {fanout} fan-out groups.");
        return fixture + parity + numeric + contracts + real + fanout;
    }
    private static int Fixtures(string repo)
    {
        var fs = JsonNode.Parse(File.ReadAllText(Path.Combine(repo, "tests/fixtures/fc-cases.json")))!["fixtures"]!.AsArray();
        var orderOutputs = new List<float[]>();
        foreach (var f in fs.Select(n => n!))
        {
            int n = f["cases"]!.GetValue<int>(), outputs = f["outputs"]!.GetValue<int>();
            int[] widths = f["widths"]!.AsArray().Select(v => v!.GetValue<int>()).ToArray();
            var g = new FcGeometry(f["name"]!.GetValue<string>(), widths, outputs);
            var x = widths.Select((k, i) => Read(n, k, f["inputs"]![i]!)).ToArray();
            var w = widths.Select((k, i) => Read(k, outputs, f["weights"]![i]!)).ToArray();
            var b = Read(1, outputs, f["bias"]!); var dy = Read(n, outputs, f["upstream"]!);
            var sx = x.Select(Values).ToArray(); var sw = w.Select(Values).ToArray(); var sb = Values(b); var sd = Values(dy);
            foreach (bool fast in new[] { false, true })
            {
                var e = new FullyConnectedAdapter(fast, g); var y = M(n, outputs); y.Fill(float.NaN);
                e.Forward(x, w, b, y); Check(f["output"]!, y, g.Name + " forward");
                if (!fast && g.Name.StartsWith("hs-order", StringComparison.Ordinal)) orderOutputs.Add(Values(y));
                var manual = M(n, outputs); manual.Fill(float.NaN);
                for (int i = 0; i < widths.Length; i++) e.ForwardInput(i, x[i], w[i], b, manual, i == 0 ? 0 : 1);
                Bits(Values(y), manual, "explicit fan-in");
                manual.Fill(float.NaN); e.ForwardInput(0, x[0], w[0], b, manual, 1);
                TestAssert.True(Values(manual).All(float.IsNaN), "forward retains NaN at scale 1");
                for (int i = 0; i < widths.Length; i++)
                {
                    var dx = M(n, widths[i]); dx.Fill(float.NaN); e.Input(i, dy, w[i], dx);
                    Check(f["input_gradients"]![i]!, dx, "fixture dx");
                    dx.Fill(3); e.Input(i, dy, w[i], dx, 1); Check(f["input_gradients"]![i]!, dx, "fixture dx add", 3);
                    dx.Fill(float.NaN); e.Input(i, dy, w[i], dx, 1);
                    TestAssert.True(Values(dx).All(float.IsNaN), "dx retains NaN");
                    var dw = M(widths[i], outputs); dw.Fill(3);
                    e.Weight(i, x[i], dy, dw, F(f["alpha"]!), F(f["beta"]!)); Check(f["weight_gradients"]![i]!, dw, "fixture alpha/beta dw");
                    dw.Fill(float.NaN); e.Weight(i, x[i], dy, dw, 1, 0);
                    var clean = M(widths[i], outputs); e.Weight(i, x[i], dy, clean); Bits(Values(clean), dw, "dw discards NaN");
                    dw.Fill(float.NaN); e.Weight(i, x[i], dy, dw, 1, 1);
                    TestAssert.True(Values(dw).All(float.IsNaN), "dw retains NaN");
                }
                var db = M(1, outputs); db.Fill(3); e.Bias(dy, db, F(f["alpha"]!), F(f["beta"]!));
                Check(f["bias_gradient"]!, db, "fixture alpha/beta db");
                db.Fill(float.NaN); e.Bias(dy, db); var cleanDb = M(1, outputs); e.Bias(dy, cleanDb);
                Bits(Values(cleanDb), db, "db discards NaN");
                db.Fill(float.NaN); e.Bias(dy, db, 1, 1); TestAssert.True(Values(db).All(float.IsNaN), "db retains NaN");
                for (int i = 0; i < x.Length; i++) { Bits(sx[i], x[i], "preserve X"); Bits(sw[i], w[i], "preserve W"); }
                Bits(sb, b, "preserve bias"); Bits(sd, dy, "preserve dY");
            }
        }
        TestAssert.True(!orderOutputs[0].SequenceEqual(orderOutputs[1]), "reverse order changes fixed W pairing");
        return fs.Count * 2;
    }
    private static int Parity()
    {
        int[] ns = [1, 2, 3, 4, 5, Vector<float>.Count, Vector<float>.Count + 1, 17, 128];
        for (int seed = 0; seed < 36; seed++)
        {
            var rng = new Random(91000 + seed); int n = ns[seed % ns.Length], o = rng.Next(1, 22);
            int[] widths = Enumerable.Range(0, seed % 3 + 1).Select(_ => rng.Next(1, 24)).ToArray();
            var g = new FcGeometry("parity", widths, o); var scalar = new FullyConnectedAdapter(false, g);
            var x = widths.Select(k => M(n, k)).ToArray(); var w = widths.Select(k => M(k, o)).ToArray();
            var b = M(1, o); var dy = M(n, o);
            foreach (var m in x.Concat(w).Append(b).Append(dy)) Fill(m, rng);
            if (seed % 4 == 0) { x[0][0, 0] = float.NaN; w[0][0, 0] = float.PositiveInfinity; dy[0, 0] = -0f; }
            if (seed % 5 == 0) { dy[0, o - 1] = float.NegativeInfinity; x[0][n - 1, 0] = -0f; }
            var sy = M(n, o); scalar.Forward(x, w, b, sy);
            foreach (int threads in new[] { 1, 2, Environment.ProcessorCount, Environment.ProcessorCount })
            {
                var fast = new FullyConnectedAdapter(true, g, threads); var y = M(n, o); y.Fill(float.NaN);
                fast.Forward(x, w, b, y); Bits(Values(sy), y, "parallel forward");
                foreach (float beta in new[] { 0, -0f, 1, .5f, -1 })
                {
                    var a = M(n, o); var z = M(n, o); a.Fill(3); z.Fill(3);
                    scalar.ForwardInput(0, x[0], w[0], b, a, beta); fast.ForwardInput(0, x[0], w[0], b, z, beta);
                    Bits(Values(a), z, "forward scale");
                    for (int i = 0; i < widths.Length; i++)
                    {
                        var sa = M(n, widths[i]); var fa = M(n, widths[i]); sa.Fill(3); fa.Fill(3);
                        scalar.Input(i, dy, w[i], sa, beta); fast.Input(i, dy, w[i], fa, beta); Bits(Values(sa), fa, "parallel dx");
                        foreach (float alpha in new[] { .3f, -.75f, 0, -0f })
                        {
                            var sd = M(widths[i], o); var fd = M(widths[i], o); sd.Fill(3); fd.Fill(3);
                            scalar.Weight(i, x[i], dy, sd, alpha, beta); fast.Weight(i, x[i], dy, fd, alpha, beta);
                            Bits(Values(sd), fd, "parallel dw");
                        }
                    }
                    foreach (float alpha in new[] { .3f, -.75f, 0, -0f })
                    {
                        var sd = M(1, o); var fd = M(1, o); sd.Fill(3); fd.Fill(3);
                        scalar.Bias(dy, sd, alpha, beta); fast.Bias(dy, fd, alpha, beta); Bits(Values(sd), fd, "parallel db");
                    }
                }
            }
        }
        return 36;
    }
    private static int Numerical()
    {
        int comparisons = 0; double worst = 0;
        foreach (int seed in new[] { 92001, 92002, 92003 })
        {
            var rng = new Random(seed); int n = seed % 3 + 2, o = 5; int[] widths = [3, 7];
            var g = new FcGeometry("numeric", widths, o);
            var xs = widths.Select(k => M(n, k)).ToArray(); var ws = widths.Select(k => M(k, o)).ToArray();
            var b = M(1, o); var dy = M(n, o);
            foreach (var m in xs.Concat(ws).Append(b).Append(dy)) Fill(m, rng);
            // Mutable double copies preserve finite-difference perturbations without float32 rounding.
            var xd = xs.Select(m => Values(m).Select(v => (double)v).ToArray()).ToArray();
            var wd = ws.Select(m => Values(m).Select(v => (double)v).ToArray()).ToArray();
            var bd = Values(b).Select(v => (double)v).ToArray(); var ud = Values(dy);
            double Loss()
            {
                double loss = 0;
                for (int c = 0; c < n; c++) for (int j = 0; j < o; j++)
                {
                    double y = bd[j];
                    for (int i = 0; i < widths.Length; i++) for (int k = 0; k < widths[i]; k++)
                        y += xd[i][k * n + c] * wd[i][j * widths[i] + k];
                    loss += y * ud[j * n + c];
                }
                return loss;
            }
            foreach (bool fast in new[] { false, true })
            {
                var e = new FullyConnectedAdapter(fast, g);
                void Verify(double[] parameter, FloatMatrix grad)
                {
                    var a = Values(grad);
                    for (int p = 0; p < parameter.Length; p++)
                    {
                        double old = parameter[p]; parameter[p] = old + 1e-5; double plus = Loss();
                        parameter[p] = old - 1e-5; double minus = Loss(); parameter[p] = old;
                        double numeric = (plus - minus) / 2e-5, error = Math.Abs(numeric - a[p]);
                        TestAssert.True(double.IsFinite(error) && error <= 2e-6 * (1 + Math.Abs(numeric)), "FC finite differences");
                        worst = Math.Max(worst, error); comparisons++;
                    }
                }
                for (int i = 0; i < widths.Length; i++)
                {
                    var dx = M(n, widths[i]); e.Input(i, dy, ws[i], dx); Verify(xd[i], dx);
                    var dw = M(widths[i], o); e.Weight(i, xs[i], dy, dw); Verify(wd[i], dw);
                }
                var db = M(1, o); e.Bias(dy, db); Verify(bd, db);
            }
        }
        Console.WriteLine($"FC NUMERICAL: {comparisons} comparisons, worst abs error {worst:G9}, tolerance 2e-6*(1+abs(numeric)), epsilon 1e-5.");
        return 6;
    }
    private static int Contracts(string original)
    {
        foreach (bool fast in new[] { false, true })
        {
            var g = new FcGeometry("contracts", [3, 5], 9); var e = new FullyConnectedAdapter(fast, g);
            var x = new[] { M(3, 3), M(3, 5) }; var w = new[] { M(3, 9), M(5, 9) }; var b = M(1, 9); var y = M(3, 9);
            y.Fill(41);
            Reject(() => e.Forward(x, [w[0], M(4, 9)], b, y), "contracts");
            TestAssert.True(Values(y).All(v => v == 41), "validate entire fan-in before writing");
            Reject(() => e.Forward(x, [w[0]], b, y), "contracts");
            Reject(() => e.ForwardInput(0, new FloatMatrix(3, 3), w[0], b, y), "F-order");
            Reject(() => e.Input(0, M(3, 9), w[0], x[0].TransposeView()), "F-order");
            Reject(() => e.Weight(0, x[0], y, w[0].TransposeView()), "F-order");
            Reject(() => e.Bias(y, M(9, 1)), "F-order");
            Reject(() => e.Input(-1, y, w[0], x[0]), "input");
            Reject(() => e.ForwardInput(0, x[0], w[0], b, x[0]), "F-order");
            // Square geometry permits exact and partial aliases to reach the storage guard.
            var ae = new FullyConnectedAdapter(fast, new("alias", [3], 3)); var square = M(3, 3); var ab = M(1, 3);
            Reject(() => ae.Forward([square], [square], ab, square), "storage");
            Reject(() => ae.Input(0, square, square, square), "storage");
            Reject(() => ae.Weight(0, square, square, square), "storage");
            var backing = M(6, 6);
            Reject(() => ae.Input(0, backing.Slice(0, 3, 0, 3), square, backing.Slice(3, 3, 3, 3)), "storage");
            Reject(() => ae.Bias(square, square.Slice(0, 1, 0, 3)), "storage");
            // Strided F views, offsets, tails and preserved sentinels, all four kernels.
            int n = 17; var rng = new Random(93001);
            FloatMatrix View(int r, int c) { var parent = M(r + 3, c + 2); parent.Fill(71); return parent.Slice(1, r, 1, c); }
            var vx = View(n, 3); var vw = View(3, 9); var vb = View(1, 9); var vy = View(n, 9);
            Fill(vx, rng); Fill(vw, rng); Fill(vb, rng); var vd = View(n, 9); Fill(vd, rng);
            var se = new FullyConnectedAdapter(false, new("strided", [3], 9)); var fe = new FullyConnectedAdapter(fast, new("strided", [3], 9));
            var ey = M(n, 9); se.Forward([vx], [vw], vb, ey); fe.Forward([vx], [vw], vb, vy); Bits(Values(ey), vy, "strided forward");
            var vdx = View(n, 3); var edx = M(n, 3); se.Input(0, vd, vw, edx); fe.Input(0, vd, vw, vdx); Bits(Values(edx), vdx, "strided dx");
            var vdw = View(3, 9); var edw = M(3, 9); se.Weight(0, vx, vd, edw, .3f); fe.Weight(0, vx, vd, vdw, .3f); Bits(Values(edw), vdw, "strided dw");
            var vdb = View(1, 9); var edb = M(1, 9); se.Bias(vd, edb, .3f); fe.Bias(vd, vdb, .3f); Bits(Values(edb), vdb, "strided db");
            foreach (var v in new[] { vy, vdx, vdw, vdb })
            {
                var occupied = Enumerable.Range(0, v.Cols).SelectMany(c => Enumerable.Range(0, v.Rows).Select(r => v.GetFlatOffset(r, c))).ToHashSet();
                for (int j = 0; j < v.Storage.Length; j++) if (!occupied.Contains(j)) TestAssert.Equal(71f, v.Storage.Span[j], "strided sentinel");
            }
            // Literal rounding witness: bias after product 0, before product 1.
            var re = new FullyConnectedAdapter(fast, new("rounding", [1, 1], 1));
            var rx = new[] { M(1, 1), M(1, 1) }; var rw = new[] { M(1, 1), M(1, 1) }; var rb = M(1, 1); var ry = M(1, 1);
            rx[0][0, 0] = 16777216; rx[1][0, 0] = -16777216; rw[0][0, 0] = rw[1][0, 0] = 1; rb[0, 0] = 1;
            re.Forward(rx, rw, rb, ry); Bits([0], ry, "bias must precede second product");
            rx[0][0, 0] = -2; rx[1][0, 0] = -3; re.Forward(rx, rw, rb, ry); Bits([-4], ry, "FC applies no ReLU");
            // +0 accumulation seed and post-reduction alpha preserve the specified zero signs.
            var ze = new FullyConnectedAdapter(fast, new("zero", [1], 1)); var zx = M(1, 1); var zd = M(1, 1); var zw = M(1, 1);
            zx[0, 0] = -0f; zd[0, 0] = 1;
            var zy = M(1, 1); var zb = M(1, 1); zb[0, 0] = -0f;
            ze.Forward([zx], [zd], zb, zy); Bits([0], zy, "forward signed zero reduction seed");
            ze.Input(0, zx, zd, zy); Bits([0], zy, "dx signed zero reduction seed");
            ze.Weight(0, zx, zd, zw, -1); Bits([-0f], zw, "negative scaled zero dw");
            ze.Bias(zx, zw, -1); Bits([-0f], zw, "negative scaled zero db");
            zx[0, 0] = float.NaN; ze.Weight(0, zx, zd, zw, 0); Bits([0], zw, "GEMM alpha zero skips NaN");
            ze.Bias(zx, zw, 0); TestAssert.True(float.IsNaN(zw[0, 0]), "sum alpha zero retains NaN");
        }
        Reject(() => new FcGeometry("bad", [], 3), "bad");
        Reject(() => new FcGeometry("bad", [0], 3), "bad");
        Reject(() => new OptimizedFullyConnected(new("bad", [1], 1), 0), "maxDegreeOfParallelism");
        var net = ShapeTests.Parse145(original); var shapes = ShapeInference.Infer(net); var layer = net["fc2048bb"]; var p = (FcLayerParams)layer.TypedParams;
        foreach (bool fast in new[] { false, true })
        {
            foreach (bool raw in new[] { false, true })
            {
                var sp = raw ? p with { WeightSource = new[] { "fc2048ba", "" } } : p with { WeightSourceLayers = new[] { "fc2048ba", "" } };
                var shared = new LayerDefinition(layer.Name, layer.Type, layer.Kind, layer.DeclaredInputs, layer.Inputs,
                    layer.SourceLayer, layer.DetachedNeuronLayer, layer.RequiresParams, sp, layer.Parameters);
                Reject(() => { if (fast) _ = OptimizedFullyConnected.FromLayer(shared, shapes[layer.Name]); else _ = ScalarFullyConnected.FromLayer(shared, shapes[layer.Name]); }, "unsupported in this port");
                Reject(() => FcGeometry.FromLayer(shared, shapes[layer.Name]), "fc2048bb");
            }
        }
        Reject(() => FcGeometry.FromLayer(layer, shapes[layer.Name] with { Inputs = new[] { "hs1a", "hs1b" } }), "disagree");
        return 7;
    }

    private static int Real(string original, bool full)
    {
        var net = ShapeTests.Parse145(original); var shapes = ShapeInference.Infer(net); int count = 0;
        string[][] inputs = [["pool3a_neuron", "pool3b_neuron"], ["pool3a_neuron", "pool3b_neuron"], ["hs1a", "hs1b"], ["hs1b", "hs1a"], ["hs2a", "hs2b"]];
        string[] names = ["fc2048a", "fc2048b", "fc2048ba", "fc2048bb", "fc1000"];
        for (int j = 0; j < names.Length; j++)
        {
            var layer = net[names[j]]; var shape = shapes[layer.Name]; int k = j < 2 ? 4608 : 2048, o = j == 4 ? 1000 : 2048;
            TestAssert.SequenceEqual(inputs[j], layer.Inputs, "real FC input order");
            TestAssert.SequenceEqual(new[] { k, k }, shape.NumInputs, "real FC widths"); TestAssert.Equal(o, shape.Outputs, "real outputs");
            var p = (FcLayerParams)layer.TypedParams; TestAssert.Equal(j == 4 ? 0d : 1d, p.InitB, "initB");
            if (j < 4)
            {
                TestAssert.Equal(layer.Name + "_neuron", layer.DetachedNeuronLayer, "separate FC neuron");
                var neuron = net[layer.DetachedNeuronLayer!]; TestAssert.Equal("relu", ((NeuronLayerParams)neuron.TypedParams).NeuronType, "FC ReLU");
                TestAssert.SequenceEqual(new[] { layer.Name }, neuron.Inputs, "FC neuron follows linear output");
                string hs = new[] { "hs1a", "hs1b", "hs2a", "hs2b" }[j];
                TestAssert.SequenceEqual(new[] { neuron.Name }, net[hs].Inputs, "hs follows separate neuron");
            }
            else { TestAssert.True(layer.DetachedNeuronLayer == null, "logits have no neuron"); TestAssert.SequenceEqual(new[] { layer.Name }, net["probs"].Inputs, "softmax follows logits"); }
            foreach (int n in full ? new[] { 3, 9 } : new[] { 3 })
            {
                var g = FcGeometry.FromLayer(layer, shape); var rng = new Random(94000 + j * 10 + n);
                var x = new[] { M(n, k), M(n, k) }; var w = new[] { M(k, o), M(k, o) }; var b = M(1, o); var dy = M(n, o);
                foreach (var m in x.Concat(w).Append(b).Append(dy)) Fill(m, rng);
                var scalar = ScalarFullyConnected.FromLayer(layer, shape); var sy = M(n, o);
                var sdx = new FloatMatrix[2]; var sdxAdded = new FloatMatrix[2]; var sdw = new FloatMatrix[2];
                if (full)
                {
                    scalar.Forward(x, w, b, sy);
                    for (int i = 0; i < 2; i++)
                    {
                        sdx[i] = M(n, k); scalar.BackwardInput(i, dy, w[i], sdx[i]);
                        sdxAdded[i] = M(n, k); sdxAdded[i].Add(sdx[i], 0); scalar.BackwardInput(i, dy, w[i], sdxAdded[i], 1);
                        sdw[i] = M(k, o); sdw[i].Fill(.125f); scalar.BackwardWeights(i, x[i], dy, sdw[i], .3f, .7f);
                    }
                }
                foreach (int threads in full ? new[] { 1, 2, Environment.ProcessorCount, Environment.ProcessorCount } : new[] { Environment.ProcessorCount })
                {
                    var e = OptimizedFullyConnected.FromLayer(layer, shape, threads); var y = M(n, o); y.Fill(float.NaN); e.Forward(x, w, b, y);
                    if (full) Bits(Values(sy), y, "real full forward");
                    for (int output = 0; output < o; output += Math.Max(1, o / 3))
                        for (int c = 0; c < n; c++)
                        {
                            float first = 0, second = 0;
                            for (int t = 0; t < k; t++) first += x[0][c, t] * w[0][t, output];
                            first += b[0, output];
                            for (int t = 0; t < k; t++) second += x[1][c, t] * w[1][t, output];
                            TestAssert.Equal(BitConverter.SingleToInt32Bits(first + second), BitConverter.SingleToInt32Bits(y[c, output]), "sampled scalar forward");
                        }
                    TestAssert.True(Values(y).All(float.IsFinite), "real finite outputs");
                    for (int i = 0; i < 2; i++)
                    {
                        var dx = M(n, k); dx.Fill(float.NaN); e.BackwardInput(i, dy, w[i], dx);
                        if (full)
                        {
                            Bits(Values(sdx[i]), dx, "real full dx");
                            e.BackwardInput(i, dy, w[i], dx, 1); Bits(Values(sdxAdded[i]), dx, "real full dx addition");
                        }
                        var dw = M(k, o); dw.Fill(.125f); e.BackwardWeights(i, x[i], dy, dw, .3f, .7f);
                        if (full)
                        {
                            Bits(Values(sdw[i]), dw, "real full dw alpha/beta");
                        }
                        for (int t = 0; t < k; t += k / 3)
                            for (int c = 0; c < n; c++)
                            {
                                float sum = 0; for (int output = 0; output < o; output++) sum += dy[c, output] * w[i][t, output];
                                if (full) sum = sum + sum;
                                TestAssert.Equal(BitConverter.SingleToInt32Bits(sum), BitConverter.SingleToInt32Bits(dx[c, t]), "sampled scalar dx");
                            }
                        for (int t = 0; t < k; t += k / 3)
                            for (int output = 0; output < o; output += o / 3)
                            {
                                float sum = 0; for (int c = 0; c < n; c++) sum += x[i][c, t] * dy[c, output];
                                float expected = .7f * .125f + .3f * sum;
                                TestAssert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(dw[t, output]), "sampled scalar dw");
                            }
                        TestAssert.True(Values(dx).All(float.IsFinite) && Values(dw).All(float.IsFinite), "real finite gradients");
                    }
                    var db = M(1, o); db.Fill(.125f); e.BackwardBiases(dy, db, .3f, .7f);
                    var sdb = M(1, o); sdb.Fill(.125f); scalar.BackwardBiases(dy, sdb, .3f, .7f); Bits(Values(sdb), db, "real db");
                }
                count++;
            }
        }
        return count;
    }

    private static int Fanout(string repo, string original)
    {
        var f = JsonNode.Parse(File.ReadAllText(Path.Combine(repo, "tests/fixtures/fc-cases.json")))!["fanout"]!;
        foreach (bool fast in new[] { false, true })
        {
            var e = new FullyConnectedAdapter(fast, new("shared", [2], 2));
            var da = Read(1, 2, f["upstream_a"]!); var db = Read(1, 2, f["upstream_b"]!);
            var wa = Read(2, 2, f["weights_a"]!); var wb = Read(2, 2, f["weights_b"]!);
            var dx = M(1, 2); dx.Fill(float.NaN); e.Input(0, da, wa, dx); Check(f["local_a"]!, dx, "fan-out first overwrites");
            e.Input(0, db, wb, dx, 1); Check(f["merged"]!, dx, "fan-out later adds");
            dx.Fill(float.NaN); e.Input(0, db, wb, dx); e.Input(0, da, wa, dx, 1); Check(f["merged"]!, dx, "fan-out reverse arrival");
        }
        var net = ShapeTests.Parse145(original); var shapes = ShapeInference.Infer(net);
        foreach (string producer in new[] { "pool3a_neuron", "pool3b_neuron", "hs1a", "hs1b" })
        {
            string[] consumers = producer.StartsWith("pool3", StringComparison.Ordinal) ? ["fc2048a", "fc2048b"] : ["fc2048ba", "fc2048bb"];
            var actual = net.Layers.Where(l => l.Inputs.Contains(producer)).Select(l => l.Name).ToArray();
            TestAssert.SequenceEqual(consumers, actual, "complete FC fan-out");
            int k = shapes[producer].Outputs, n = 3; var contributions = new FloatMatrix[2];
            for (int i = 0; i < 2; i++)
            {
                var layer = net[consumers[i]]; var g = FcGeometry.FromLayer(layer, shapes[layer.Name]);
                int slot = layer.Inputs.ToList().IndexOf(producer); var dy = M(n, g.Outputs); var w = M(k, g.Outputs);
                var rng = new Random(95000 + i); Fill(dy, rng); Fill(w, rng);
                contributions[i] = M(n, k); contributions[i].Fill(float.NaN);
                var e = new OptimizedFullyConnected(g); e.BackwardInput(slot, dy, w, contributions[i]);
                for (int t = 0; t < k; t += k / 3) for (int c = 0; c < n; c++)
                {
                    float sum = 0; for (int o = 0; o < g.Outputs; o++) sum += dy[c, o] * w[t, o];
                    TestAssert.Equal(sum, contributions[i][c, t], "fan-out scalar contribution");
                }
            }
            int ownGpu = net[producer].TypedParams.Gpu;
            int local = Array.FindIndex(consumers, c => net[c].TypedParams.Gpu == ownGpu), remote = 1 - local;
            foreach (int[] arrivals in new[] { new[] { 0, 1 }, new[] { 1, 0 } })
            {
                var buffers = new[] { M(n, k), M(n, k) }; foreach (var m in buffers) m.Fill(float.NaN);
                foreach (int arrival in arrivals) buffers[arrival].Add(contributions[arrival], 0, 1);
                // Original src/layer.cu:178,202-212,228-251: existing local then remote.
                buffers[local].Add(buffers[remote], 1, 1);
                var expected = M(n, k); expected.Add(contributions[local], 0, 1); expected.Add(contributions[remote], 1, 1);
                Bits(Values(expected), buffers[local], "local-first remote merge independent of arrival");
            }
        }
        TestAssert.SequenceEqual(new[] { "fc1000" }, net.Layers.Where(l => l.Inputs.Contains("hs2b")).Select(l => l.Name).ToArray(), "hs2b remote-only consumer");
        return 6;
    }
}
