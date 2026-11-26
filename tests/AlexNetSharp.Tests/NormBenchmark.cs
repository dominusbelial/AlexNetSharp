using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using AlexNetSharp.Core.Layers.Norm;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class NormBenchmark
{
    public static int Run(string original, string[] args)
    {
#if DEBUG
        throw new InvalidOperationException("Use Release for the normalisation benchmark.");
#else
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        var network = ShapeTests.Parse145(original); var shapes = ShapeInference.Infer(network);
        var rows = new List<object>();
        Console.WriteLine($"NORM BENCHMARK: N=128, threads={Environment.ProcessorCount}, SIMD={Vector<float>.Count}; 3 warmups + 5 samples/op; median ms, allocations included.");
        Console.WriteLine("| Layer | Scalar forward | Optimized forward | Scalar backward | Optimized backward |");
        Console.WriteLine("| --- | ---: | ---: | ---: | ---: |");
        foreach (var layer in network.Layers.Where(l => l.Type is "cmrnorm" or "rnorm"))
        {
            var shape = shapes[layer.Name]; var fast = OptimizedNorm.FromLayer(layer, shape); var scalar = ScalarNorm.FromLayer(layer, shape);
            var x = new FloatMatrix(shape.Outputs, 128); var y = new FloatMatrix(x.Rows, 128); var d = new FloatMatrix(x.Rows, 128);
            var dy = new FloatMatrix(x.Rows, 128); var dx = new FloatMatrix(x.Rows, 128);
            NormTests.Fill(x, new Random(88001)); NormTests.Fill(dy, new Random(88002));
            var sf = Time(() => scalar.Forward(x, d, y)); var ff = Time(() => fast.Forward(x, d, y));
            var sb = Time(() => scalar.Backward(x, dy, d, y, dx)); var fb = Time(() => fast.Backward(x, dy, d, y, dx));
            Console.WriteLine($"| {layer.Name} | {Median(sf):F3} | {Median(ff):F3} | {Median(sb):F3} | {Median(fb):F3} |");
            rows.Add(new { layer = layer.Name, scalar_forward = sf, optimized_forward = ff, scalar_backward = sb, optimized_backward = fb });
        }
        int index = Array.IndexOf(args, "--norm-output");
        if (index >= 0)
        {
            if (index + 1 >= args.Length) throw new ArgumentException("--norm-output needs a path");
            File.WriteAllText(args[index + 1], JsonSerializer.Serialize(new { cases = 128, threads = Environment.ProcessorCount, simd = Vector<float>.Count, rows },
                new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }
        return 0;
#endif
    }
    private static double[] Time(Action action)
    {
        for (int i = 0; i < 3; i++) action();
        var samples = new double[5];
        for (int i = 0; i < samples.Length; i++) { var sw = Stopwatch.StartNew(); action(); samples[i] = sw.Elapsed.TotalMilliseconds; }
        return samples;
    }
    private static double Median(double[] samples) => samples.Order().ElementAt(samples.Length / 2);
}
