using System.Diagnostics;
using System.Text.Json;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Layers.Dropout;
using AlexNetSharp.Core.Layers.Softmax;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class DropoutSoftmaxCostBenchmark
{
    public static int Run(string original, string[] args)
    {
#if DEBUG
        throw new InvalidOperationException("Use Release for the dropout/softmax/cost benchmark.");
#else
        int at = Array.IndexOf(args, "--dsc-threads");
        int threads = at < 0 ? Environment.ProcessorCount : int.Parse(args[at + 1]);
        if (threads <= 0) throw new ArgumentOutOfRangeException(nameof(threads));
        var net = ShapeTests.Parse145(original); var rows = new List<object>();
        var rng = new Random(100500);
        Console.WriteLine($"ANX-10 BENCHMARK: batch=128, threads={threads}; 2 warmups + 3 samples, median ms; preallocated operands/targets, internal allocation included.");
        Console.WriteLine("| Layer/op | Scalar ms | Optimized ms |"); Console.WriteLine("| --- | ---: | ---: |");
        void Measure(string name, Action scalar, Action optimized)
        {
            var s = Time(scalar); var f = Time(optimized);
            Console.WriteLine($"| {name} | {s.Order().ElementAt(1):F3} | {f.Order().ElementAt(1):F3} |");
            rows.Add(new { name, scalar_ms = s, optimized_ms = f });
        }
        foreach (string name in new[] { "hs1a", "hs1b", "hs2a", "hs2b" })
        {
            var p = (DropoutLayerParams)net[name].TypedParams;
            var x = DropoutSoftmaxCostTests.C(2048, 128); var y = DropoutSoftmaxCostTests.C(2048, 128); var dy = DropoutSoftmaxCostTests.C(2048, 128); var dx = DropoutSoftmaxCostTests.C(2048, 128);
            DropoutSoftmaxCostTests.Fill(x, rng); DropoutSoftmaxCostTests.Fill(dy, rng);
            var scalar = new ScalarDropout((float)p.Keep, p.Enable, new SeededDropoutMaskSource(100501));
            var fast = new OptimizedDropout((float)p.Keep, p.Enable, new SeededDropoutMaskSource(100501), threads);
            Measure(name + " train forward (RNG included)", () => scalar.Forward(x, y), () => fast.Forward(x, y));
            Measure(name + " backward", () => scalar.Backward(dy, dx), () => fast.Backward(dy, dx));
            Measure(name + " eval", () => scalar.Forward(x, y, DropoutMode.Evaluate), () => fast.Forward(x, y, DropoutMode.Evaluate));
        }
        var logits = DropoutSoftmaxCostTests.S(128, 1000); var probs = DropoutSoftmaxCostTests.S(128, 1000);
        var upstream = DropoutSoftmaxCostTests.S(128, 1000); var grad = DropoutSoftmaxCostTests.S(128, 1000); var labels = DropoutSoftmaxCostTests.C(1, 128);
        DropoutSoftmaxCostTests.Fill(logits, rng); DropoutSoftmaxCostTests.Fill(upstream, rng);
        Measure("probs forward", () => DropoutSoftmaxCostTests.Forward(false, logits, probs), () => DropoutSoftmaxCostTests.Forward(true, logits, probs, threads));
        Measure("probs general backward", () => DropoutSoftmaxCostTests.Backward(false, probs, upstream, grad), () => DropoutSoftmaxCostTests.Backward(true, probs, upstream, grad, 0, threads));
        Measure("probs fused backward", () => DropoutSoftmaxCostTests.Fused(false, labels, probs, grad), () => DropoutSoftmaxCostTests.Fused(true, labels, probs, grad, 1, 0, threads));
        Measure("logprob forward top5", () => DropoutSoftmaxCostTests.Cost(false, labels, probs.TransposeView(), 5), () => DropoutSoftmaxCostTests.Cost(true, labels, probs.TransposeView(), 5, threads));
        Measure("logprob probability backward", () => DropoutSoftmaxCostTests.ProbabilityBackward(false, labels, probs.TransposeView(), grad.TransposeView()), () => DropoutSoftmaxCostTests.ProbabilityBackward(true, labels, probs.TransposeView(), grad.TransposeView(), 1, 0, threads));
        var ms = new AlexNetSharp.Core.Layers.Cost.LogregMultiviewAccumulator(); var mf = new AlexNetSharp.Core.Layers.Cost.LogregMultiviewAccumulator(true, threads);
        Measure("logprob ten-view accumulation + cost", () => Views(ms), () => Views(mf));
        void Views(AlexNetSharp.Core.Layers.Cost.LogregMultiviewAccumulator a)
        {
            for (int v = 0; v < 10; v++) a.Forward(labels, probs.TransposeView(), 5, v == 0 ? AlexNetSharp.Core.Layers.Cost.MultiviewPhase.Start : v == 9 ? AlexNetSharp.Core.Layers.Cost.MultiviewPhase.End : AlexNetSharp.Core.Layers.Cost.MultiviewPhase.Continue);
        }
        int output = Array.IndexOf(args, "--dsc-output");
        if (output >= 0) File.WriteAllText(args[output + 1], JsonSerializer.Serialize(new { batch = 128, threads, repeats = 3, warmups = 2, rows }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        return 0;
#endif
    }
    private static double[] Time(Action action)
    {
        action(); action(); var samples = new double[3];
        for (int i = 0; i < samples.Length; i++) { var sw = Stopwatch.StartNew(); action(); samples[i] = sw.Elapsed.TotalMilliseconds; }
        return samples;
    }
}
