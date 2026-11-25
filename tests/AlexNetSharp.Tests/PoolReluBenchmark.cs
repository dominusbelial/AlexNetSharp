using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using AlexNetSharp.Core.Layers.Pool;
using AlexNetSharp.Core.Layers.Neurons;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class PoolReluBenchmark
{
    public static int Run(string original,string[] args)
    {
#if DEBUG
        throw new InvalidOperationException("Use Release for the pooling benchmark.");
#else
        CultureInfo.CurrentCulture=CultureInfo.InvariantCulture;
        var network=ShapeTests.Parse145(original);var shapes=ShapeInference.Infer(network);
        var rows=new List<object>();
        Console.WriteLine($"POOL BENCHMARK: N=128, threads={Environment.ProcessorCount}, SIMD={Vector<float>.Count}, 3 warmups + 5 samples/op, median ms.");
        Console.WriteLine("| Layer | Scalar fwd | SIMD fwd | Scalar back | SIMD back | ReLU fwd | ReLU back |");
        Console.WriteLine("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
        foreach(var layer in network.Layers.Where(l=>l.Type=="pool"))
        {
            var shape=shapes[layer.Name];var p=OptimizedPool.FromLayer(layer,shape);var scalar=ScalarPool.FromLayer(layer,shape);
            var x=new FloatMatrix(p.Geometry.InputRows,128);var y=new FloatMatrix(p.Geometry.OutputRows,128);
            var dy=new FloatMatrix(y.Rows,128);var dx=new FloatMatrix(x.Rows,128);var ry=new FloatMatrix(y.Rows,128);var rd=new FloatMatrix(y.Rows,128);
            PoolReluTests.Fill(x,new Random(7801));PoolReluTests.Fill(dy,new Random(7802));
            double[] sf=Time(()=>scalar.Forward(x,y)),ff=Time(()=>p.Forward(x,y));
            double[] sb=Time(()=>scalar.Backward(x,dy,y,dx)),fb=Time(()=>p.Backward(x,dy,y,dx));
            bool hasRelu=network.Layers.Any(l=>l.Name==layer.Name+"_neuron");
            double[]? rf=hasRelu?Time(()=>OptimizedRelu.Forward(y,ry)):null;
            double[]? rb=hasRelu?Time(()=>OptimizedRelu.Backward(ry,dy,rd)):null;
            Console.WriteLine($"| {layer.Name} | {Median(sf):F3} | {Median(ff):F3} | {Median(sb):F3} | {Median(fb):F3} | {(rf==null?"—":Median(rf).ToString("F3"))} | {(rb==null?"—":Median(rb).ToString("F3"))} |");
            rows.Add(new {layer=layer.Name,scalar_forward=sf,optimized_forward=ff,scalar_backward=sb,optimized_backward=fb,relu_forward=rf,relu_backward=rb});
        }
        int index=Array.IndexOf(args,"--pool-output");
        if(index>=0)
        {
            if(index+1>=args.Length)throw new ArgumentException("--pool-output needs a path");
            File.WriteAllText(args[index+1],JsonSerializer.Serialize(new{cases=128,threads=Environment.ProcessorCount,simd=Vector<float>.Count,rows},new JsonSerializerOptions{WriteIndented=true})+"\n");
        }
        return 0;
#endif
    }
    private static double[] Time(Action action)
    {
        for(int i=0;i<3;i++)action();var result=new double[5];for(int i=0;i<5;i++){var sw=Stopwatch.StartNew();action();result[i]=sw.Elapsed.TotalMilliseconds;}return result;
    }
    private static double Median(double[] samples)=>samples.Order().ElementAt(samples.Length/2);
}
