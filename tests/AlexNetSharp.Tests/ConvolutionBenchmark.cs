using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using AlexNetSharp.Core.Layers.Conv;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

// Explicit opt-in mode. Timed calls include packing, allocation, scheduling and final stores.
internal static class ConvolutionBenchmark
{
    private sealed record Row(string Layer, int Cases, int Threads, string Operation, double Milliseconds,
        double Flops, double Gflops, long WorkspaceBytes, double[] Samples);

    public static int Run(string origRoot, string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        int threads = OptionInt("--conv-threads", Environment.ProcessorCount);
        int repeats = OptionInt("--conv-repeats", 3);
        string? output = Option("--conv-output");
        var net = ShapeTests.Parse145(origRoot); var shapes = ShapeInference.Infer(net);
        var rows = new List<Row>(); var small = new List<object>();
        Console.WriteLine($"CONV BENCHMARK: CPU={CpuModel()}, logical cores={Environment.ProcessorCount}, threads={threads}, SIMD lanes={Vector<float>.Count}, repeats={repeats} (median), one warmup/op.");
#if DEBUG
        if (!args.Contains("--allow-debug-benchmark")) throw new InvalidOperationException("Use Release for the benchmark.");
#endif
        Console.WriteLine("| Layer | Forward ms / GF/s | Input ms / GF/s | Weights ms / GF/s | Bias ms / GF/s | Peak workspace MiB |");
        Console.WriteLine("| --- | ---: | ---: | ---: | ---: | ---: |");
        foreach (var layer in net.Layers.Where(l => l.Type == "conv"))
        {
            var shape = shapes[layer.Name]; var c = OptimizedConvolution.FromLayer(layer, shape, threads);
            var s = ScalarConvolution.FromLayer(layer, shape);
            bool dxEnabled = !layer.Name.StartsWith("conv1", StringComparison.Ordinal);
            var rng = new Random(66000 + rows.Count);
            MeasureBatch(128, false);
            MeasureBatch(2, true);
            var large = rows.Where(r => r.Layer == layer.Name && r.Cases == 128).ToArray();
            string Cell(string op) { var r = large.SingleOrDefault(r => r.Operation == op); return r is null ? "omitted (data)" : $"{r.Milliseconds:F3} / {r.Gflops:F2}"; }
            Console.WriteLine($"| {layer.Name} | {Cell("forward")} | {Cell("input")} | {Cell("weights")} | {Cell("bias")} | {large.Max(r => r.WorkspaceBytes) / 1048576.0:F3} |");

            void MeasureBatch(int cases, bool compare)
            {
                var xs = c.Inputs.Select(g => OptimizedConvolutionTests.RandomMatrix(g.Channels * g.ImageSize * g.ImageSize, cases, rng)).ToArray();
                var ws = c.Inputs.Select(g => OptimizedConvolutionTests.RandomMatrix(g.Channels * g.FilterSize * g.FilterSize, c.Filters, rng)).ToArray();
                var b = OptimizedConvolutionTests.RandomMatrix(c.SharedBiases ? c.Filters : c.Filters * c.Modules, 1, rng);
                var y = new FloatMatrix(c.Filters * c.Modules, cases);
                var dy = OptimizedConvolutionTests.RandomMatrix(y.Rows, cases, rng);
                var dx = xs.Select(x => new FloatMatrix(x.Rows, cases)).ToArray();
                var dw = ws.Select(w => new FloatMatrix(w.Rows, w.Cols)).ToArray(); var db = new FloatMatrix(b.Rows, 1);
                double dense = c.Inputs.Sum(g => 2.0 * cases * c.Modules * c.Filters * g.Channels * g.FilterSize * g.FilterSize);
                double inputFlops = c.Inputs.Sum(g => InputFlops(g, c.Filters, cases));
                var operations = new List<(string Name, Action Fast, Action Slow, double Flops, long Workspace)>
                {
                    ("forward", () => c.Forward(xs, ws, b, y), () => s.Forward(xs, ws, b, y), dense,
                        c.Inputs.Max(g => 4L * g.Channels * g.FilterSize * g.FilterSize * OptimizedConvolution.ForwardTile * Math.Min(threads, (c.Modules * cases - 1) / OptimizedConvolution.ForwardTile + 1))),
                    ("weights", () => { for (int i = 0; i < xs.Length; i++) c.BackwardWeights(i, xs[i], dy, dw[i]); },
                        () => { for (int i = 0; i < xs.Length; i++) s.BackwardWeights(i, xs[i], dy, dw[i]); }, dense,
                        c.Inputs.Max(g => 4L * (Math.Min(OptimizedConvolution.ReductionTile, c.Modules * cases) * c.Filters + 2 * g.Channels * g.FilterSize * g.FilterSize * c.Filters + Math.Min(threads, g.Channels * g.FilterSize * g.FilterSize) * Math.Min(OptimizedConvolution.ReductionTile, c.Modules * cases)))),
                    ("bias", () => c.BackwardBiases(dy, db), () => s.BackwardBiases(dy, db), (double)y.ElementCount, 0)
                };
                if (dxEnabled) operations.Insert(1, ("input", () => { for (int i = 0; i < xs.Length; i++) c.BackwardInput(i, dy, ws[i], dx[i]); },
                    () => { for (int i = 0; i < xs.Length; i++) s.BackwardInput(i, dy, ws[i], dx[i]); }, inputFlops, 0));
                foreach (var op in operations)
                {
                    var samples = Time(op.Fast, repeats); double ms = Median(samples);
                    rows.Add(new(layer.Name, cases, threads, op.Name, ms, op.Flops, op.Flops / (ms * 1e6), op.Workspace, samples));
                    if (compare)
                    {
                        var scalarSamples = Time(op.Slow, repeats); double scalarMs = Median(scalarSamples);
                        small.Add(new { Layer = layer.Name, Operation = op.Name, Cases = cases, ScalarMs = scalarMs,
                            OptimizedMs = ms, Speedup = scalarMs / ms, ScalarSamples = scalarSamples });
                    }
                }
            }
        }
        double convMs = rows.Where(r => r.Cases == 128).Sum(r => r.Milliseconds);
        double convFlops = rows.Where(r => r.Cases == 128).Sum(r => r.Flops);
        double rate = convFlops / (convMs * 1e6);
        double projectedStepMs = 4.3 * 128 / rate * 1000;
        double convDays = convMs / 1000 * (1_200_000.0 * 90 / 128) / 86400;
        double totalDays = projectedStepMs / 1000 * (1_200_000.0 * 90 / 128) / 86400;
        Console.WriteLine($"CONV PROJECTION: batch128 conv={convMs:F3} ms, {convFlops / 1e9:F3} GFLOP, aggregate={rate:F2} GFLOP/s; conv-only 90 epochs={convDays:F2} days. DESIGN 4.3 GFLOP/image extrapolation: step={projectedStepMs:F3} ms, run={totalDays:F2} days; conv FLOP share={convFlops / (4.3e9 * 128):P2}.");
        Console.WriteLine("SMALL BATCH SCALAR SPEEDUPS (JSON): " + JsonSerializer.Serialize(small));
        if (output is not null) File.WriteAllText(output, JsonSerializer.Serialize(new { Cpu = CpuModel(), Threads = threads,
            SimdLanes = Vector<float>.Count, Repeats = repeats, Rows = rows, SmallBatch = small,
            ConvMilliseconds = convMs, ConvGflops = rate, ConvDays = convDays, ProjectedStepMs = projectedStepMs, ProjectedDays = totalDays }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        return 0;

        string? Option(string key) { int i = Array.IndexOf(args, key); return i >= 0 && i + 1 < args.Length ? args[i + 1] : null; }
        int OptionInt(string key, int fallback) { int n = int.Parse(Option(key) ?? fallback.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture); ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n); return n; }
    }

    private static double InputFlops(ConvInputGeometry g, int filters, int cases)
    {
        long taps = 0;
        for (int my = 0; my < g.ModulesX; my++) for (int mx = 0; mx < g.ModulesX; mx++)
            for (int fy = 0; fy < g.FilterSize; fy++) for (int fx = 0; fx < g.FilterSize; fx++)
            {
                int y = g.PaddingStart + my * g.Stride + fy, x = g.PaddingStart + mx * g.Stride + fx;
                if (y >= 0 && y < g.ImageSize && x >= 0 && x < g.ImageSize) taps++;
            }
        return 2.0 * taps * g.Channels * filters * cases;
    }
    private static double[] Time(Action action, int repeats)
    {
        action();
        var samples = new double[repeats];
        for (int i = 0; i < repeats; i++) { var sw = Stopwatch.StartNew(); action(); sw.Stop(); samples[i] = sw.Elapsed.TotalMilliseconds; }
        return samples;
    }
    private static double Median(double[] samples) { var sorted = samples.Order().ToArray(); return sorted[sorted.Length / 2]; }
    private static string CpuModel() => File.Exists("/proc/cpuinfo")
        ? File.ReadLines("/proc/cpuinfo").First(l => l.StartsWith("model name", StringComparison.Ordinal)).Split(':', 2)[1].Trim()
        : System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();
}
