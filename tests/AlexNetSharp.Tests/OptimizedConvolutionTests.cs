using AlexNetSharp.Core.Layers.Conv;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class OptimizedConvolutionTests
{
    public static int RunAll(string origRoot, bool fullScalar)
    {
        int count = 0;
        for (int test = 0; test < 66; test++)
        {
            var rng = new Random(61000 + test);
            int kernel = test % 11 + 1, stride = Math.Min(kernel, test % 4 + 1);
            int size = kernel + rng.Next(1, 9), padding = -(test % 3);
            var g = new ConvInputGeometry(size, test % 5 + 1, kernel, stride, padding);
            // Every third layer has a second input with distinct geometry and the same module count.
            var gs = test % 3 == 0 ? new[] { g, new ConvInputGeometry(g.ModulesX, 2, 1, 1) } : new[] { g };
            int partial = test % 2 == 0 ? 0 : g.ModulesX;
            Verify(gs, test % 13 + 1, test % 5 + 1, partial, test % 4 != 0, rng, true);
            count++;
        }
        // SIMD input-gradient lanes, reduction panel boundaries, partial chunks straddling panels,
        // and multiple forward tiles all have explicit coverage beyond the 1..5-case random set.
        Verify([new(65, 1, 1, 1)], 11, 3, 65, true, new(69001), true);
        Verify([new(7, 3, 3, 2, -1)], 13, 17, 0, false, new(69002), true);
        Verify([new(9, 2, 3, 2)], 9, 128, 4, true, new(69003), true);
        count += 3;
        if (fullScalar)
        {
            var net = ShapeTests.Parse145(origRoot); var shapes = ShapeInference.Infer(net);
            foreach (var layer in net.Layers.Where(l => l.Type == "conv"))
            {
                var c = ScalarConvolution.FromLayer(layer, shapes[layer.Name]);
                // All four operations, both conv3 inputs; conv1 data gradient remains caller-omitted.
                Verify(c.Inputs, c.Filters, 2, c.PartialSum, c.SharedBiases,
                    new Random(69500 + count), false, !layer.Name.StartsWith("conv1", StringComparison.Ordinal));
                Console.WriteLine($"OPTIMIZED REAL SCALAR PARITY PASSED: {layer.Name}, N=2, every element, zero tolerance.");
                count++;
            }
        }
        Validation(); count++;
        Console.WriteLine($"OPTIMIZED PARITY/DETERMINISM PASSED: {count} groups; scalar bits, threads 1/2/{Environment.ProcessorCount}, repeated max; fixed seeds; tolerance 0.");
        return count;
    }

    private static void Verify(IReadOnlyList<ConvInputGeometry> gs, int filters, int cases, int partial,
        bool sharedBiases, Random rng, bool exhaustive, bool inputGradient = true)
    {
        var xs = gs.Select(g => RandomMatrix(g.Channels * g.ImageSize * g.ImageSize, cases, rng)).ToArray();
        var ws = gs.Select(g => RandomMatrix(g.Channels * g.FilterSize * g.FilterSize, filters, rng)).ToArray();
        int outputs = filters * gs[0].ModulesX * gs[0].ModulesX;
        var b = RandomMatrix(sharedBiases ? filters : outputs, 1, rng);
        var dy = RandomMatrix(outputs, cases, rng);
        ConvolutionAdapter.UseOptimized = false;
        var oracle = Execute(new("parity", gs, filters, partial, sharedBiases));
        foreach (int threads in new[] { 1, 2, Environment.ProcessorCount, Environment.ProcessorCount })
        {
            ConvolutionAdapter.UseOptimized = true;
            var actual = Execute(new("parity", gs, filters, partial, sharedBiases, threads));
            TestAssert.Equal(oracle.Count, actual.Count, "parity operation count");
            for (int operation = 0; operation < oracle.Count; operation++)
            {
                var expected = oracle[operation]; var got = actual[operation];
                for (int i = 0; i < expected.Length; i++)
                    if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(got[i]))
                        throw new TestFailureException($"Raw parity: threads={threads}, op={operation}, index={i}, expected={expected[i]:G9}, got={got[i]:G9}.");
            }
            // Real oracle parity is expensive; only thread-count comparisons are repeated thereafter.
            if (!exhaustive) oracle = actual;
        }

        List<float[]> Execute(ConvolutionAdapter conv)
        {
            var result = new List<float[]>();
            void Add(FloatMatrix m) => result.Add(m.Storage.ToArray());
            var y = new FloatMatrix(outputs, cases); y.Fill(float.NaN);
            conv.Forward(xs, ws, b, y); Add(y);
            foreach (float targetScale in exhaustive ? new[] { 0f, 1f } : new[] { 0f })
            {
                for (int i = 0; i < gs.Count; i++)
                {
                    if (exhaustive)
                    {
                        y.Fill(targetScale == 0 ? float.NaN : .125f);
                        conv.ForwardInput(i, xs[i], ws[i], b, y, targetScale); Add(y);
                    }
                    if (inputGradient)
                    {
                        var dx = new FloatMatrix(xs[i].Rows, cases); dx.Fill(targetScale == 0 ? float.NaN : .125f);
                        conv.BackwardInput(i, dy, ws[i], dx, targetScale); Add(dx);
                    }
                    foreach (bool gc in exhaustive ? new[] { false, true } : new[] { false })
                    {
                        var dw = new FloatMatrix(ws[i].Rows, filters); dw.Fill(targetScale == 0 ? float.NaN : .125f);
                        conv.BackwardWeights(i, xs[i], dy, dw, targetScale, gc, exhaustive ? cases * 2 : null); Add(dw);
                    }
                }
                foreach (bool gc in exhaustive ? new[] { false, true } : new[] { false })
                {
                    var db = new FloatMatrix(b.Rows, 1); db.Fill(targetScale == 0 ? float.NaN : .125f);
                    conv.BackwardBiases(dy, db, targetScale, gc, exhaustive ? cases * 2 : null); Add(db);
                }
            }
            if (exhaustive)
            {
                y.Fill(float.NaN); conv.ForwardInput(0, xs[0], ws[0], b, y, 1);
                TestAssert.True(y.Storage.Span.ToArray().All(float.IsNaN), "scale1 forward NaNs retained");
                var dx = new FloatMatrix(xs[0].Rows, cases); dx.Fill(float.NaN); conv.BackwardInput(0, dy, ws[0], dx, 1);
                TestAssert.True(dx.Storage.Span.ToArray().All(float.IsNaN), "scale1 dx NaNs retained");
                var dw = new FloatMatrix(ws[0].Rows, filters); dw.Fill(float.NaN); conv.BackwardWeights(0, xs[0], dy, dw, 1);
                TestAssert.True(dw.Storage.Span.ToArray().All(float.IsNaN), "scale1 dw NaNs retained");
                var db = new FloatMatrix(b.Rows, 1); db.Fill(float.NaN); conv.BackwardBiases(dy, db, 1);
                TestAssert.True(db.Storage.Span.ToArray().All(float.IsNaN), "scale1 db NaNs retained");
                Add(y); Add(dx); Add(dw); Add(db);
            }
            return result;
        }
    }

    private static void Validation()
    {
        // Direct construction exercises optimized validation independently of the ANX-5 adapter.
        foreach (var g in new[] { new ConvInputGeometry(3, 2, 1, 1, Groups: 2), new ConvInputGeometry(3, 2, 1, 1, RandSparse: true) })
        {
            try { _ = new OptimizedConvolution("named-rejection", [g], 2); throw new TestFailureException("Missing rejection"); }
            catch (NotSupportedException e) { TestAssert.True(e.Message.Contains("named-rejection", StringComparison.Ordinal), "named rejection"); }
        }
        var o = new OptimizedConvolution("atomic-validation", [new(3, 1, 1, 1), new(3, 1, 1, 1)], 1);
        var x = new FloatMatrix(9, 1); var w = new FloatMatrix(1, 1); var b = new FloatMatrix(1, 1); var y = new FloatMatrix(9, 1); y.Fill(77);
        try { o.Forward([x, x], [w, new(2, 1)], b, y); throw new TestFailureException("Missing shape rejection"); }
        catch (ArgumentException) { TestAssert.True(y.Storage.ToArray().All(v => v == 77), "fan-in validates before any write"); }
        try { _ = new OptimizedConvolution("threads", [new(3, 1, 1, 1)], 1, maxDegreeOfParallelism: 0); throw new TestFailureException("Missing threads rejection"); }
        catch (ArgumentOutOfRangeException) { }
    }

    internal static FloatMatrix RandomMatrix(int rows, int cols, Random rng)
    {
        var m = new FloatMatrix(rows, cols);
        for (int i = 0; i < m.ElementCount; i++) m.Storage.Span[i] = (float)((rng.NextDouble() - .5) * .25);
        return m;
    }
}
