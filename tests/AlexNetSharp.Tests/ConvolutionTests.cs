using System.Diagnostics;
using System.Text.Json.Nodes;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Layers.Conv;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class ConvolutionTests
{
    public static int RunAll(string repoRoot, string origRoot)
    {
        int count = Fixtures(repoRoot);
        count += NumericalGradients();
        count += Contracts();
        count += RealGeometry(origRoot);
        return count;
    }

    private static int Fixtures(string repoRoot)
    {
        var fixtures = JsonNode.Parse(File.ReadAllText(Path.Combine(repoRoot, "tests/fixtures/conv-cases.json")))!["cases"]!.AsArray();
        int count = 0;
        foreach (var node in fixtures)
        {
            var c = node!;
            string name = c["name"]!.GetValue<string>();
            int filters = Int(c, "filters"), cases = Int(c, "cases"), width = Int(c, "modules_x");
            var ins = c["inputs"]!.AsArray().Select(v => v!).ToArray();
            var geometry = ins.Select(i => new ConvInputGeometry(Int(i, "size"), Int(i, "channels"),
                Int(i, "filter"), Int(i, "stride"), -Int(i, "padding"))).ToArray();
            var images = ins.Select((i, n) => Matrix(geometry[n].Channels * geometry[n].ImageSize * geometry[n].ImageSize, cases, i["images"]!)).ToArray();
            var weights = ins.Select((i, n) => Matrix(geometry[n].Channels * geometry[n].FilterSize * geometry[n].FilterSize, filters, i["weights"]!)).ToArray();
            var bias = Matrix(filters, 1, c["biases"]!);
            var dy = Matrix(filters * width * width, cases, c["dy"]!);
            foreach (int partial in new[] { 0, Int(c, "partial_sum") })
            {
                var conv = new ScalarConvolution(name, geometry, filters, partial);
                TestAssert.Equal(width, conv.ModulesX, name + " ceiling width");
                var y = new FloatMatrix(dy.Rows, cases);
                y.Fill(float.NaN);
                conv.Forward(images, weights, bias, y);
                Check(c["forward"]!, y, name + " forward");
                // Every fixture proves first-write bias, subsequent input fan-in and scale 0/1.
                var fan = new FloatMatrix(dy.Rows, cases);
                fan.Fill(float.NaN);
                for (int i = 0; i < ins.Length; i++) conv.ForwardInput(i, images[i], weights[i], bias, fan, i == 0 ? 0 : 1);
                Check(c["forward"]!, fan, name + " explicit fan-in");
                fan.Fill(2);
                conv.ForwardInput(0, images[0], weights[0], bias, fan, 1);
                var first = new FloatMatrix(dy.Rows, cases);
                conv.ForwardInput(0, images[0], weights[0], bias, first);
                for (int r = 0; r < fan.Rows; r++)
                    for (int n = 0; n < cases; n++)
                        Near(first[r, n] - bias[r / (width * width), 0] + 2, fan[r, n], name + " forward scale1", 0);
                for (int i = 0; i < ins.Length; i++)
                {
                    var dx = new FloatMatrix(images[i].Rows, cases);
                    dx.Fill(float.NaN);
                    conv.BackwardInput(i, dy, weights[i], dx);
                    Check(ins[i]["dx"]!, dx, name + " dx");
                    conv.BackwardInput(i, dy, weights[i], dx, 1);
                    Check(ins[i]["dx"]!, dx, name + " dx add", 2);
                    foreach (bool gc in new[] { false, true })
                    {
                        var dw = new FloatMatrix(weights[i].Rows, filters);
                        dw.Fill(float.NaN);
                        conv.BackwardWeights(i, images[i], dy, dw, gradientCheck: gc);
                        var expected = ins[i][gc ? "dw" : "dw_training"]!;
                        Check(expected, dw, name + " dw");
                        conv.BackwardWeights(i, images[i], dy, dw, 1, gc);
                        Check(expected, dw, name + " dw add", 2);
                    }
                }
                foreach (bool gc in new[] { false, true })
                {
                    var db = new FloatMatrix(filters, 1);
                    db.Fill(float.NaN);
                    conv.BackwardBiases(dy, db, gradientCheck: gc);
                    var expected = c[gc ? "db" : "db_training"]!;
                    Check(expected, db, name + " db");
                    conv.BackwardBiases(dy, db, 1, gc);
                    Check(expected, db, name + " db add", 2);
                }
                count++;
            }
        }
        Console.WriteLine($"CONV FIXTURES PASSED: {fixtures.Count} independent rational cases, partialSum 0 and divisor, all four operations, training/GC and overwrite/add.");
        return count;
    }

    private static int NumericalGradients()
    {
        var sets = new[]
        {
            new[] { new ConvInputGeometry(6, 2, 3, 2) },
            new[] { new ConvInputGeometry(4, 1, 3, 2, -1) },
            new[] { new ConvInputGeometry(16, 1, 11, 4) },
            new[] { new ConvInputGeometry(4, 1, 3, 2), new ConvInputGeometry(2, 2, 1, 1) },
        };
        int checks = 0, count = 0;
        double worst = 0;
        for (int test = 0; test < sets.Length; test++)
        {
            var gs = sets[test];
            int cases = test == 1 ? 3 : 2, filters = test == 2 ? 1 : 2;
            var rng = new Random(7301 + test);
            var xs = gs.Select(g => RandomMatrix(g.Channels * g.ImageSize * g.ImageSize, cases, rng)).ToArray();
            var ws = gs.Select(g => RandomMatrix(g.Channels * g.FilterSize * g.FilterSize, filters, rng)).ToArray();
            var b = RandomMatrix(filters, 1, rng);
            int width = gs[0].ModulesX;
            var dy = RandomMatrix(filters * width * width, cases, rng);
            // Float64 independent objective operates on copies, never on float32-perturbed values.
            var xd = xs.Select(m => m.Storage.ToArray().Select(v => (double)v).ToArray()).ToArray();
            var wd = ws.Select(m => m.Storage.ToArray().Select(v => (double)v).ToArray()).ToArray();
            var bd = b.Storage.ToArray().Select(v => (double)v).ToArray();
            foreach (int partial in new[] { 0, width })
                foreach (bool gc in new[] { true, false })
                {
                    var conv = new ScalarConvolution("numeric", gs, filters, partial);
                    double scale = gc ? 1 : 1.0 / cases;
                    double Loss() => Objective(gs, filters, cases, xd, wd, bd, dy) * scale;
                    for (int i = 0; i < gs.Length; i++)
                    {
                        var dx = new FloatMatrix(xs[i].Rows, cases);
                        var dw = new FloatMatrix(ws[i].Rows, filters);
                        conv.BackwardInput(i, dy, ws[i], dx);
                        conv.BackwardWeights(i, xs[i], dy, dw, gradientCheck: gc);
                        Differentiate(xd[i], dx.Storage.ToArray().Select(v => v * scale).ToArray(), Loss, ref checks, ref worst);
                        Differentiate(wd[i], dw.Storage.ToArray().Select(v => (double)v).ToArray(), Loss, ref checks, ref worst);
                    }
                    var db = new FloatMatrix(filters, 1);
                    conv.BackwardBiases(dy, db, gradientCheck: gc);
                    Differentiate(bd, db.Storage.ToArray().Select(v => (double)v).ToArray(), Loss, ref checks, ref worst);
                    count++;
                }
        }
        Console.WriteLine($"CONV NUMERICAL PASSED: {checks} central float64 differences, epsilon=1e-5, tolerance=2e-5*(1+abs(expected)), worst absolute error={worst:G6}, seeds 7301..7304.");
        return count;
    }

    // Independently traverse input pixels, test membership in output windows and scatter in float64.
    // No call to the implementation under test and no float32 finite-difference perturbation.
    private static double Objective(ConvInputGeometry[] gs, int filters, int cases,
        double[][] xs, double[][] ws, double[] biases, FloatMatrix dy)
    {
        int width = gs[0].ModulesX, modules = width * width;
        var y = new double[filters * modules * cases];
        for (int f = 0; f < filters; f++)
            for (int m = 0; m < modules; m++)
                for (int n = 0; n < cases; n++) y[(f * modules + m) * cases + n] = biases[f];
        for (int i = 0; i < gs.Length; i++)
        {
            var g = gs[i];
            for (int feature = 0; feature < g.Channels * g.ImageSize * g.ImageSize; feature++)
            {
                int c = feature / (g.ImageSize * g.ImageSize);
                int iy = feature / g.ImageSize % g.ImageSize, ix = feature % g.ImageSize;
                for (int m = 0; m < modules; m++)
                {
                    int ky = iy - g.PaddingStart - m / width * g.Stride;
                    int kx = ix - g.PaddingStart - m % width * g.Stride;
                    if (ky < 0 || ky >= g.FilterSize || kx < 0 || kx >= g.FilterSize) continue;
                    int k = (c * g.FilterSize + ky) * g.FilterSize + kx;
                    for (int f = 0; f < filters; f++)
                        for (int n = 0; n < cases; n++)
                            y[(f * modules + m) * cases + n] += xs[i][feature * cases + n] * ws[i][k * filters + f];
                }
            }
        }
        double loss = 0;
        for (int r = 0; r < dy.Rows; r++)
            for (int n = 0; n < cases; n++) loss += y[r * cases + n] * dy[r, n];
        return loss;
    }

    private static void Differentiate(double[] variables, double[] gradient, Func<double> loss,
        ref int checks, ref double worst)
    {
        const double epsilon = 1e-5;
        for (int j = 0; j < variables.Length; j++)
        {
            double old = variables[j];
            variables[j] = old + epsilon;
            double plus = loss();
            variables[j] = old - epsilon;
            double minus = loss();
            variables[j] = old;
            double numeric = (plus - minus) / (2 * epsilon);
            worst = Math.Max(worst, Math.Abs(numeric - gradient[j]));
            Near(numeric, gradient[j], "numerical gradient", 2e-5);
            checks++;
        }
    }

    private static int Contracts()
    {
        // Cancellation makes a chunk boundary observably different from the flat reduction.
        var x = new FloatMatrix(9, 1);
        x.Storage.Span[0] = 1e8f; x.Storage.Span[1] = 1;
        x.Storage.Span[2] = -1e8f; x.Storage.Span[3] = 1;
        var dy = new FloatMatrix(9, 1); dy.Fill(1);
        var flat = new ScalarConvolution("flat", [new(3, 1, 1, 1)], 1);
        var chunks = new ScalarConvolution("chunks", [new(3, 1, 1, 1)], 1, 3);
        var a = new FloatMatrix(1, 1); var b = new FloatMatrix(1, 1);
        flat.BackwardWeights(0, x, dy, a, gradientCheck: true);
        chunks.BackwardWeights(0, x, dy, b, gradientCheck: true);
        Near(1, a[0, 0], "flat association", 0); Near(1, b[0, 0], "chunk association", 0);
        // Use [1e8,1,1,-1e8,1,1,0,0,0]: flat=2, chunks=0.
        x.Fill(0); x.Storage.Span[0] = 1e8f; x.Storage.Span[1] = 1; x.Storage.Span[2] = 1;
        x.Storage.Span[3] = -1e8f; x.Storage.Span[4] = 1; x.Storage.Span[5] = 1;
        flat.BackwardWeights(0, x, dy, a, gradientCheck: true);
        chunks.BackwardWeights(0, x, dy, b, gradientCheck: true);
        Near(2, a[0, 0], "flat cancellation", 0); Near(0, b[0, 0], "chunk cancellation", 0);
        // Three cases: scaling must happen within chunks, before their final sum.
        var three = new FloatMatrix(9, 3); var d3 = new FloatMatrix(9, 3); d3.Fill(1);
        three[0, 0] = 3; three[3, 0] = 6; three[6, 0] = 9;
        chunks.BackwardWeights(0, three, d3, b, normalizationCases: 6);
        Near(3, b[0, 0], "virtual batch weight normalization", 0);
        var db = new FloatMatrix(1, 1);
        chunks.BackwardBiases(d3, db, normalizationCases: 6);
        Near(4.5, db[0, 0], "virtual batch bias normalization", 0);
        three.Fill(0); three[0, 0] = 1e8f; three[3, 0] = -1e8f + 8; three[6, 0] = 1;
        chunks.BackwardWeights(0, three, d3, b);
        Near(2.3333332538604736, b[0, 0], "scale chunks before sum", 0);
        flat.BackwardWeights(0, three, d3, a);
        Near(3, a[0, 0], "scale full reduction", 0);


        var conv = new ScalarConvolution("unshared", [new(3, 1, 1, 1)], 1, sharedBiases: false);
        var bias = new FloatMatrix(9, 1);
        for (int r = 0; r < 9; r++) bias[r, 0] = r;
        var w = new FloatMatrix(1, 1); w[0, 0] = 2;
        var y = new FloatMatrix(9, 3); three.Fill(1);
        conv.Forward([three], [w], bias, y);
        for (int r = 0; r < 9; r++) for (int n = 0; n < 3; n++) Near(r + 2, y[r, n], "unshared bias", 0);
        conv.BackwardBiases(d3, bias);
        for (int r = 0; r < 9; r++) Near(1, bias[r, 0], "unshared db", 0);

        Expect<NotSupportedException>(() => new ScalarConvolution("group-layer", [new(3, 2, 1, 1, Groups: 2)], 2), "group-layer");
        Expect<NotSupportedException>(() => new ScalarConvolution("sparse-layer", [new(3, 2, 1, 1, RandSparse: true)], 2), "sparse-layer");
        Expect<ArgumentException>(() => new ScalarConvolution("bad-partial", [new(3, 1, 1, 1)], 1, 2), "partialSum");
        Expect<ArgumentException>(() => flat.ForwardInput(0, x, w, new FloatMatrix(1, 1), x), "storage");
        Expect<ArgumentException>(() => flat.BackwardInput(0, dy, w, dy), "storage");
        Expect<ArgumentException>(() => flat.BackwardWeights(0, x, dy, x), "canonical");
        Expect<ArgumentException>(() => flat.BackwardBiases(dy, dy), "canonical");
        Expect<ArgumentException>(() => flat.ForwardInput(0, new FloatMatrix(9, 1, MatrixLayout.ColumnMajor), w, new FloatMatrix(1, 1), dy), "canonical");
        Expect<ArgumentOutOfRangeException>(() => flat.BackwardBiases(dy, new FloatMatrix(1, 1), normalizationCases: 0), "count");
        // Strided canonical ANX-4 slices retain row strides and storage offsets.
        var xp = new FloatMatrix(11, 3); xp.Fill(91);
        var wp = new FloatMatrix(3, 3); wp.Fill(92);
        var yp = new FloatMatrix(11, 3); yp.Fill(93);
        var bp = new FloatMatrix(3, 3); bp.Fill(94);
        var xv = xp.Slice(1, 9, 1, 1); xv.Fill(2);
        var wv = wp.Slice(1, 1, 1, 1); wv[0, 0] = 3;
        var yv = yp.Slice(1, 9, 1, 1); var bv = bp.Slice(1, 1, 1, 1); bv[0, 0] = 4;
        flat.ForwardInput(0, xv, wv, bv, yv);
        for (int r = 0; r < 9; r++) Near(10, yv[r, 0], "strided forward", 0);
        flat.BackwardInput(0, yv, wv, xv);
        for (int r = 0; r < 9; r++) Near(30, xv[r, 0], "strided dx", 0);
        flat.BackwardWeights(0, xv, yv, wv, gradientCheck: true);
        Near(2700, wv[0, 0], "strided dw", 0);
        flat.BackwardBiases(yv, bv, gradientCheck: true); Near(90, bv[0, 0], "strided db", 0);
        Near(91, xp[0, 0], "input sentinel", 0); Near(92, wp[0, 0], "weight sentinel", 0);
        Near(93, yp[0, 0], "output sentinel", 0); Near(94, bp[0, 0], "bias sentinel", 0);
        return 5;
    }

    private static int RealGeometry(string origRoot)
    {
        var net = ShapeTests.Parse145(origRoot);
        var shapes = ShapeInference.Infer(net);
        int count = 0;
        foreach (var layer in net.Layers.Where(l => l.Type == "conv"))
        {
            var shape = shapes[layer.Name];
            var conv = ScalarConvolution.FromLayer(layer, shape);
            var p = LayerParameterBuffers.Allocate(shape);
            const int cases = 2;
            var rng = new Random(14500 + count);
            var xs = shape.NumInputs.Select(rows => RandomMatrix(rows, cases, rng)).ToArray();
            var ws = p.Weights.Select(w => w.Matrix).ToArray();
            foreach (var w in ws) FillRandom(w, rng);
            FillRandom(p.Biases!, rng);
            var y = new FloatMatrix(shape.Outputs, cases);
            var dy = RandomMatrix(shape.Outputs, cases, rng);
            var sw = Stopwatch.StartNew();
            conv.Forward(xs, ws, p.Biases!, y);
            long forward = sw.ElapsedMilliseconds;
            var inputGrads = new List<FloatMatrix>();
            // Original table omits conv1's data gradient. Later layers' adjoints run here.
            bool inputGradient = !layer.Name.StartsWith("conv1", StringComparison.Ordinal);
            for (int i = 0; i < xs.Length; i++)
            {
                if (inputGradient)
                {
                    var dx = new FloatMatrix(xs[i].Rows, cases);
                    dx.Fill(float.NaN);
                    conv.BackwardInput(i, dy, ws[i], dx);
                    Finite(dx, layer.Name + " dx");
                    TestAssert.Equal(shape.NumInputs[i], dx.Rows, layer.Name + " input grad rows");
                    inputGrads.Add(dx);
                }
            }
            long inputMs = sw.ElapsedMilliseconds - forward;
            for (int i = 0; i < xs.Length; i++)
            {
                var dw = new FloatMatrix(ws[i].Rows, ws[i].Cols);
                dw.Fill(float.NaN);
                conv.BackwardWeights(i, xs[i], dy, dw);
                Finite(dw, layer.Name + " dw");
                // Euler identity: dot(dW,W) = dot(dY, convolution)/N, independent of bias.
                double rhs = 0;
                var contribution = new FloatMatrix(shape.Outputs, cases);
                var zeroBias = new FloatMatrix(p.Biases!.Rows, 1);
                conv.ForwardInput(i, xs[i], ws[i], zeroBias, contribution);
                for (int r = 0; r < dy.Rows; r++) for (int n = 0; n < cases; n++) rhs += (double)dy[r, n] * contribution[r, n];
                double weightDot = Dot(dw, ws[i]) * cases;
                Near(rhs, weightDot, layer.Name + " real weight Euler", 5e-4);
                if (inputGradient) Near(rhs, Dot(inputGrads[i], xs[i]), layer.Name + " real input adjoint", 5e-4);
            }
            long weightMs = sw.ElapsedMilliseconds - forward - inputMs;
            var db = new FloatMatrix(p.Biases!.Rows, 1);
            conv.BackwardBiases(dy, db); Finite(db, layer.Name + " db");
            sw.Stop();
            Finite(y, layer.Name + " forward");
            TestAssert.Equal(shape.Outputs, y.Rows, layer.Name + " output rows");
            TestAssert.Equal(cases, y.Cols, layer.Name + " cases");
            TestAssert.Equal(shape.ModulesX, conv.ModulesX, layer.Name + " modules");
            Console.WriteLine($"CONV REAL {layer.Name}: output={conv.Filters}x{conv.ModulesX}x{conv.ModulesX}x{cases}, inputs={xs.Length}, forward={forward} ms, input={inputMs} ms{(inputGradient ? "" : " (omitted: data)")}, weights+adjoint/Euler proof={weightMs} ms, total={sw.ElapsedMilliseconds} ms.");
            count++;
        }
        TestAssert.Equal(10, count, "all ten 145 conv layers");
        Conv1Boundary(net, shapes);
        return count + 1;
    }

    private static void Conv1Boundary(NetworkDefinition net, NetworkShapes shapes)
    {
        foreach (string name in new[] { "conv1a", "conv1b" })
        {
            var conv = ScalarConvolution.FromLayer(net[name], shapes[name]);
            var g = conv.Inputs[0];
            TestAssert.Equal(216, g.PaddingStart + (conv.ModulesX - 1) * g.Stride, "conv1 final start");
            TestAssert.SequenceEqual(Enumerable.Range(216, 11).ToArray(), Enumerable.Range(0, g.FilterSize).Select(k => g.PaddingStart + (conv.ModulesX - 1) * g.Stride + k).ToArray(), "conv1 samples 216..226");
            // Actual scalar reads: encode x in channel 0, select each filter column separately.
            // At a valid y, final x window must read 216..223, then 0,0,0; repeat for y.
            var xs = new FloatMatrix(g.Channels * g.ImageSize * g.ImageSize, 1);
            for (int y = 0; y < g.ImageSize; y++) for (int x = 0; x < g.ImageSize; x++) xs[y * g.ImageSize + x, 0] = x + 1;
            var w = new FloatMatrix(g.Channels * g.FilterSize * g.FilterSize, conv.Filters);
            var b = new FloatMatrix(conv.Filters, 1); var output = new FloatMatrix(conv.Filters * conv.Modules, 1);
            for (int k = 0; k < 11; k++) w[k, k] = 1;
            conv.Forward([xs], [w], b, output);
            for (int k = 0; k < 11; k++) Near(k < 8 ? 217 + k : 0, output[k * conv.Modules + 54, 0], "conv1 actual right read", 0);
            xs.Fill(0); w.Fill(0);
            for (int y = 0; y < g.ImageSize; y++) for (int x = 0; x < g.ImageSize; x++) xs[y * g.ImageSize + x, 0] = y + 1;
            for (int k = 0; k < 11; k++) w[k * 11, k] = 1;
            conv.Forward([xs], [w], b, output);
            for (int k = 0; k < 11; k++) Near(k < 8 ? 217 + k : 0, output[k * conv.Modules + 54 * 55, 0], "conv1 actual bottom read", 0);
        }
        Console.WriteLine("CONV1 BOUNDARY PASSED: actual per-tap reads 216..223 plus 3 zeros, right and bottom, both towers.");
    }

    private static double Dot(FloatMatrix a, FloatMatrix b)
    {
        double sum = 0;
        for (int r = 0; r < a.Rows; r++) for (int n = 0; n < a.Cols; n++) sum += (double)a[r, n] * b[r, n];
        return sum;
    }
    private static void Finite(FloatMatrix m, string label)
    {
        for (int r = 0; r < m.Rows; r++) for (int n = 0; n < m.Cols; n++) TestAssert.True(float.IsFinite(m[r, n]), label + " finite");
    }
    private static int Int(JsonNode n, string key) => n[key]!.GetValue<int>();
    private static FloatMatrix Matrix(int rows, int cols, JsonNode data)
    {
        var m = new FloatMatrix(rows, cols);
        TestAssert.Equal(m.ElementCount, data.AsArray().Count, "fixture length");
        for (int i = 0; i < m.ElementCount; i++) m.Storage.Span[i] = data[i]!.GetValue<float>();
        return m;
    }
    private static void Check(JsonNode expected, FloatMatrix actual, string label, double factor = 1)
    {
        TestAssert.Equal(actual.ElementCount, expected.AsArray().Count, label + " length");
        for (int r = 0; r < actual.Rows; r++) for (int n = 0; n < actual.Cols; n++)
            Near(expected[r * actual.Cols + n]!.GetValue<double>() * factor, actual[r, n], label, 0);
    }
    private static void Near(double expected, double actual, string label, double tolerance)
    {
        if (!double.IsFinite(expected) || !double.IsFinite(actual) || Math.Abs(expected - actual) > tolerance * (1 + Math.Abs(expected)))
            throw new TestFailureException($"{label}: expected {expected:G17}, got {actual:G17}, tolerance {tolerance:G6}.");
    }
    private static FloatMatrix RandomMatrix(int rows, int cols, Random rng)
    {
        var m = new FloatMatrix(rows, cols); FillRandom(m, rng); return m;
    }
    private static void FillRandom(FloatMatrix m, Random rng)
    {
        for (int r = 0; r < m.Rows; r++) for (int n = 0; n < m.Cols; n++) m[r, n] = (float)((rng.NextDouble() - .5) * .25);
    }
    private static void Expect<T>(Action action, string message) where T : Exception
    {
        try { action(); } catch (T ex) { TestAssert.True(ex.Message.Contains(message, StringComparison.Ordinal), "exception message: " + ex.Message); return; }
        throw new TestFailureException("Expected " + typeof(T).Name + ": " + message);
    }
}
