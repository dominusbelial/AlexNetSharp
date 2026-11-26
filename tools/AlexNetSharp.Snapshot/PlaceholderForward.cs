using System.Buffers.Binary;
using System.Globalization;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;
using AlexNetSharp.Core.Layers.Conv;
using AlexNetSharp.Core.Layers.Pool;
using AlexNetSharp.Core.Layers.Neurons;
using AlexNetSharp.Core.Layers.Norm;
using AlexNetSharp.Core.Layers.FullyConnected;
using AlexNetSharp.Core.Layers.Dropout;
using AlexNetSharp.Core.Layers.Softmax;
using AlexNetSharp.Core.Layers.Cost;

using AlexNetSharp.Replay;
using StbImageSharp;

namespace AlexNetSharp.Snapshot;

/// <summary>Only a one-image replay producer: placeholder weights, not ANX-11 initialisation.</summary>
public static class PlaceholderForward
{
    public static NetworkDefinition Parse(string root) => NetworkConfigParser.ParseFiles(
        Path.Combine(root, "layers/layers-145.cfg"), Path.Combine(root, "layers/layer-params-145.cfg"));

    // Fixed SplitMix64 + Box-Muller; not System.Random, NumPy or historical CUDA RNG.
    private sealed class Gaussian(ulong seed)
    {
        private ulong state = seed;
        private double Uniform()
        {
            ulong z = state += 0x9e3779b97f4a7c15UL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
            return ((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0) + (0.5 / 9007199254740992.0);
        }
        public float Next(double std) => (float)(std * Math.Sqrt(-2 * Math.Log(Uniform())) * Math.Cos(2 * Math.PI * Uniform()));
    }
    public static float[] LoadPhoto(string root)
    {
        // Managed stb decoder is confined to the producer assembly; never downloaded by the browser.
        var image = ImageResult.FromMemory(File.ReadAllBytes(Path.Combine(root,"content/samples/labrador.jpg")), ColorComponents.RedGreenBlue);
        return Preprocess(image.Data, image.Width, image.Height);
    }
    public static float[] Preprocess(byte[] rgb, int width, int height)
    {
        if(width <= 0 || height <= 0 || rgb.Length != checked(width*height*3)) throw new InvalidDataException("Expected RGB photo.");
        double scale = 256.0 / Math.Min(width, height);
        int rw = (int)Math.Round(width * scale), rh = (int)Math.Round(height * scale);
        int left = (rw - 224) / 2, top = (rh - 224) / 2;
        var result = new float[3 * 224 * 224]; int[] means = [123, 117, 104];
        for (int c = 0; c < 3; c++) for (int y = 0; y < 224; y++) for (int x = 0; x < 224; x++)
        {
            int sx = Math.Min(width - 1, (int)((x + left) / scale)), sy = Math.Min(height - 1, (int)((y + top) / scale));
            result[(c * 224 + y) * 224 + x] = rgb[(sy * width + sx) * 3 + c] - means[c];
        }
        return result;
    }

    public sealed record ForwardResult(NetworkDefinition Definition, NetworkShapes Shapes, Dictionary<string, FloatMatrix> Activations,
        float[] Conv1Filters, double[] Cost);

    public static ForwardResult Run(NetworkDefinition definition, float[] image, ulong seed, int workers = 1)
    {
        var shapes = ShapeInference.Infer(definition); var random = new Gaussian(seed);
        var values = new Dictionary<string, FloatMatrix>(StringComparer.Ordinal);
        var filters = new List<float>(); double[] cost = [];
        FloatMatrix Execute(string name)
        {
            if (values.TryGetValue(name, out var saved)) return saved;
            var l = definition[name]; var shape = shapes[name];
            var inputs = l.Inputs.Select(Execute).ToArray();
            var output = new FloatMatrix(Math.Max(1, shape.Outputs), 1);
            if (l.Kind == LayerKind.Data)
            {
                if (((DataLayerParams)l.TypedParams).DataIdx == 0)
                {
                    if (image.Length != shape.Outputs) throw new ArgumentException("Image dimensions.");
                    image.CopyTo(output.Storage.Span);
                }
                // Synthetic label index 0; it carries no semantic class meaning.
            }
            else if (l.Kind is LayerKind.Conv or LayerKind.Fc)
            {
                var p = LayerParameterBuffers.Allocate(shape);
                var std = l.TypedParams is ConvLayerParams cp ? cp.InitW : ((FcLayerParams)l.TypedParams).InitW;
                double bias = l.TypedParams is ConvLayerParams cb ? cb.InitB : ((FcLayerParams)l.TypedParams).InitB;
                for (int i = 0; i < p.Weights.Count; i++)
                    for (int j = 0; j < p.Weights[i].Matrix.Storage.Length; j++) p.Weights[i].Matrix.Storage.Span[j] = random.Next(std[i]);
                p.Biases!.Fill((float)bias);
                if (l.Kind == LayerKind.Conv)
                {
                    OptimizedConvolution.FromLayer(l, shape, workers).Forward(inputs, p.Weights.Select(w => w.Matrix).ToArray(), p.Biases!, output);
                    if (name is "conv1a" or "conv1b")
                    {
                        var w = (ConvWeightBuffer)p.Weights[0];
                        for (int f = 0; f < 48; f++) for (int c = 0; c < 3; c++) for (int y = 0; y < 11; y++) for (int x = 0; x < 11; x++) filters.Add(w[c, y, x, f]);
                    }
                }
                else OptimizedFullyConnected.FromLayer(l, shape, workers).Forward(inputs.Select(x => x.TransposeView()).ToArray(),
                    p.Weights.Select(w => w.Matrix).ToArray(), p.Biases!, output.TransposeView());
            }
            else switch (l.Kind)
            {
                case LayerKind.Pool: OptimizedPool.FromLayer(l, shape, workers).Forward(inputs[0], output); break;
                case LayerKind.Neuron: OptimizedRelu.Forward(inputs[0].TransposeView(), output.TransposeView(), workers); break;
                case LayerKind.CmrNorm: case LayerKind.RNorm:
                    OptimizedNorm.FromLayer(l, shape, workers).Forward(inputs[0], new FloatMatrix(shape.Outputs, 1), output); break;
                case LayerKind.Hs:
                    var p = (DropoutLayerParams)l.TypedParams;
                    new OptimizedDropout((float)p.Keep, p.Enable, new SeededDropoutMaskSource(seed), workers).Forward(inputs[0], output, DropoutMode.Evaluate); break;
                case LayerKind.Softmax: OptimizedSoftmax.Forward(inputs[0].TransposeView(), output.TransposeView(), workers); break;
                case LayerKind.CostLogreg:
                    var result = OptimizedLogregCost.Forward(inputs[0], inputs[1], ((LogregCostLayerParams)l.TypedParams).TopK, workers);
                    cost = [result.NegativeLogLikelihood, result.Top1Error, result.TopKError];
                    output = new FloatMatrix(1, 1); break;
                default: throw new NotSupportedException(l.Type);
            }
            values.Add(name, output); return output;
        }
        foreach (var l in definition.Layers) Execute(l.Name);
        return new(definition, shapes, values, filters.ToArray(), cost);
    }

    public static ReplaySnapshot Produce(string root, ulong seed = 145)
    {
        byte[] image = File.ReadAllBytes(Path.Combine(root, "content/samples/labrador.jpg"));
        var run = Run(Parse(root), LoadPhoto(root), seed);
        using var payload = new MemoryStream();
        Blob Append(float[] values)
        {
            int offset = checked((int)payload.Position);
            byte[] bytes = new byte[values.Length * 4];
            for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
            payload.Write(bytes); return new("activations.bin", offset, values.Length, ReplaySnapshot.Hash(bytes));
        }
        var layers = new List<ReplayLayer>(); var dims = new Dictionary<string, int[]>();
        int[] Dimensions(string name)
        {
            if (dims.TryGetValue(name, out var d)) return d;
            var l = run.Definition[name]; var s = run.Shapes[name];
            int[] shape = l.Kind switch
            {
                LayerKind.Data => name == "data" ? [3,224,224] : [1,1,1],
                LayerKind.Conv => [((ConvLayerParams)l.TypedParams).OutputFilters,s.ModulesX!.Value,s.ModulesX.Value],
                LayerKind.Pool => [((PoolLayerParams)l.TypedParams).Channels,s.OutputsX!.Value,s.OutputsX.Value],
                LayerKind.Neuron or LayerKind.CmrNorm or LayerKind.RNorm or LayerKind.Hs => Dimensions(l.Inputs[0]).ToArray(),
                _ => [s.Outputs,1,1]
            };
            dims[name] = shape; return shape;
        }
        foreach (var l in run.Definition.Layers)
        {
            var s = run.Shapes[l.Name];
            var values = s.Outputs == 0 ? [] : run.Activations[l.Name].Storage.ToArray();
            double mean = values.Length == 0 ? 0 : values.Average(v => (double)v);
            var stats = new ActivationStats(values.Length == 0 ? 0 : values.Min(), values.Length == 0 ? 0 : values.Max(), mean,
                values.Length == 0 ? 0 : Math.Sqrt(values.Average(v => ((double)v - mean) * (v - mean))), values.Length);
            (int kernel, int stride, int start) = l.TypedParams switch
            {
                ConvLayerParams p => (p.FilterSize[0],p.Stride[0],p.PaddingStart[0]),
                PoolLayerParams p => (p.SizeX,p.Stride,p.Start),
                RNormLayerParams p => (p.Size,1,-p.Size/2),
                _ => (1,1,0)
            };
            layers.Add(new(l.Name,l.Type,Dimensions(l.Name),l.TypedParams.Gpu,l.Inputs.ToArray(),
                run.Definition.Layers.Where(c => c.Inputs.Contains(l.Name)).Select(c => c.Name).ToArray(),
                l.SourceLayer is not null ? $"relu after {l.SourceLayer}" : l.DetachedNeuronLayer is not null ? $"then {l.DetachedNeuronLayer}" : null,
                s.WeightsShapes.Sum(w => (long)w[0]*w[1]) + (s.BiasesShape.Count == 0 ? 0 : (long)s.BiasesShape[0]*s.BiasesShape[1]),
                new(l.Parameters.ToDictionary(p => p.Key,p => p.Value.ToString()),StringComparer.Ordinal),kernel,stride,start,stats,Append(values)));
        }
        var manifest = new ReplayManifest(1,0,
            ReplaySnapshot.Hash(File.ReadAllBytes(Path.Combine(root,"layers/layers-145.cfg"))),
            ReplaySnapshot.Hash(File.ReadAllBytes(Path.Combine(root,"layers/layer-params-145.cfg"))),seed,
            "placeholder, not ANX-11 initialisation; SplitMix64 Box-Muller Gaussian; configured initW/initB", "float32-le","channel,y,x","eval",
            new("content/samples/labrador.jpg",ReplaySnapshot.Hash(image),"Dktue, Portrait of a labrador retriever; content/samples/PROVENANCE.md","CC0-1.0",
                "not the ANX-12 pipeline; short side 256 nearest-neighbour, centre crop 224, RGB constant mean subtraction",[123,117,104]),
            layers.ToArray(),Append(run.Conv1Filters),run.Activations["probs"].Storage.ToArray().Select((p,i) => new RankedClass(i,p))
                .OrderByDescending(c => c.Probability).ThenBy(c => c.Index).Take(5).ToArray(),run.Cost,
            ["Run index can append immutable step manifests", "Optional curves, gradientStatistics and filter blobs in a later schema version"]);
        return new(manifest,payload.ToArray());
    }
}
