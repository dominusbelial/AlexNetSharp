using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using AlexNetSharp.Core.Layers.FullyConnected;
using AlexNetSharp.Core.Shapes;

namespace AlexNetSharp.Tests;

internal static class FullyConnectedBenchmark
{
    public static int Run(string original, string[] args)
    {
#if DEBUG
        throw new InvalidOperationException("Use Release for the FC benchmark.");
#else
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        var net = ShapeTests.Parse145(original); var shapes = ShapeInference.Infer(net); var rows = new List<object>();
        int threads = Number(args, "--fc-threads", Environment.ProcessorCount), repeats = Number(args, "--fc-repeats", 3);
        Console.WriteLine($"FC BENCHMARK: N=128, threads={threads}, SIMD={Vector<float>.Count}; 2 warmups + {repeats} samples/op, allocations included; ms / GFLOP/s.");
        Console.WriteLine("| Layer | Forward | Input gradient | Weight gradient | Bias gradient |");
        Console.WriteLine("| --- | ---: | ---: | ---: | ---: |");
        foreach (var layer in net.Layers.Where(l => l.Type == "fc"))
        {
            var shape = shapes[layer.Name]; var e = OptimizedFullyConnected.FromLayer(layer, shape, threads);
            var rng = new Random(96001); int n = 128, o = shape.Outputs;
            var x = shape.NumInputs.Select(k => FullyConnectedTests.M(n, k)).ToArray();
            var w = shape.NumInputs.Select(k => FullyConnectedTests.M(k, o)).ToArray();
            var dx = shape.NumInputs.Select(k => FullyConnectedTests.M(n, k)).ToArray();
            var dw = shape.NumInputs.Select(k => FullyConnectedTests.M(k, o)).ToArray();
            var b = FullyConnectedTests.M(1, o); var db = FullyConnectedTests.M(1, o);
            var y = FullyConnectedTests.M(n, o); var dy = FullyConnectedTests.M(n, o);
            foreach (var m in x.Concat(w).Append(b).Append(dy)) FullyConnectedTests.Fill(m, rng);
            var forward = Time(() => e.Forward(x, w, b, y), repeats);
            var input = Time(() => { for (int i = 0; i < x.Length; i++) e.BackwardInput(i, dy, w[i], dx[i]); }, repeats);
            var weight = Time(() => { for (int i = 0; i < x.Length; i++) e.BackwardWeights(i, x[i], dy, dw[i], .01f / n, .9f); }, repeats);
            var bias = Time(() => e.BackwardBiases(dy, db, 1f / n), repeats);
            double flops = 2d * n * o * shape.NumInputs.Sum(), biasFlops = (double)n * o;
            string Cell(double[] samples, double work) { double ms = Median(samples); return $"{ms:F3} / {work / ms / 1e6:F2}"; }
            Console.WriteLine($"| {layer.Name} | {Cell(forward, flops)} | {Cell(input, flops)} | {Cell(weight, flops)} | {Cell(bias, biasFlops)} |");
            rows.Add(new { layer = layer.Name, cases = n, widths = shape.NumInputs, outputs = o, inputs = shape.Inputs,
                forward_ms = forward, input_ms = input, weight_ms = weight, bias_ms = bias, product_flops = flops, bias_flops = biasFlops,
                workspace_bytes = 4L * shape.NumInputs.Max() * n });
        }
        int index = Array.IndexOf(args, "--fc-output");
        if (index >= 0)
        {
            if (index + 1 >= args.Length) throw new ArgumentException("--fc-output needs a path");
            File.WriteAllText(args[index + 1], JsonSerializer.Serialize(new { cases = 128, threads, simd = Vector<float>.Count, repeats, warmups = 2, rows },
                new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }
        return 0;
#endif
    }
    private static int Number(string[] args, string key, int fallback)
    {
        int at = Array.IndexOf(args, key);
        if (at < 0) return fallback;
        if (at + 1 >= args.Length || !int.TryParse(args[at + 1], out int result) || result <= 0) throw new ArgumentException(key + " needs a positive integer");
        return result;
    }
    private static double[] Time(Action action, int count)
    {
        for (int i = 0; i < 2; i++) action();
        var samples = new double[count];
        for (int i = 0; i < count; i++) { var sw = Stopwatch.StartNew(); action(); samples[i] = sw.Elapsed.TotalMilliseconds; }
        return samples;
    }
    private static double Median(double[] values) => values.Order().ElementAt(values.Length / 2);
}
