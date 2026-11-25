using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Pool;

/// <summary>Plain-loop float32 oracle. ANX-4 (channel*y*x,case) layout.
/// Clipped windows; max seed -2e38, numeric fmax (NaNs ignored), all ties receive
/// full gradients. Average divides by clipped area. Backward sums oy,ox in order,
/// including multiplication by a zero mask (thus 0*NaN is NaN).
/// A:include/cudaconv2/conv_util.cuh:71-94,161-186;
/// A:src/cudaconv2/conv_util.cu:1430-1470,1512-1582.</summary>
public sealed class ScalarPool
{
    public string Name { get; }
    public PoolGeometry Geometry { get; }
    public string Pool { get; }
    public ScalarPool(string name, PoolGeometry geometry, string pool = "max")
    {
        PoolContract.Geometry(name, geometry, pool);
        Name = name; Geometry = geometry; Pool = pool;
    }
    public static ScalarPool FromLayer(LayerDefinition layer, LayerShape shape) =>
        new(layer.Name, PoolContract.FromLayer(layer, shape), ((PoolLayerParams)layer.TypedParams).Pool);

    public void Forward(FloatMatrix input, FloatMatrix target)
    {
        var g = Geometry;
        PoolContract.Matrix(Name, input, g.InputRows, target.Cols);
        PoolContract.Matrix(Name, target, g.OutputRows, input.Cols);
        PoolContract.Disjoint(Name, target, input);
        for (int c = 0; c < g.Channels; c++)
            for (int oy = 0; oy < g.OutputSize; oy++)
                for (int ox = 0; ox < g.OutputSize; ox++)
                {
                    int sy = g.Start + oy * g.Stride, sx = g.Start + ox * g.Stride;
                    int ly = Math.Max(0, sy), lx = Math.Max(0, sx);
                    int ey = Math.Min(g.ImageSize, sy + g.SizeX), ex = Math.Min(g.ImageSize, sx + g.SizeX);
                    for (int n = 0; n < input.Cols; n++)
                    {
                        float value = Pool == "max" ? -2e38f : 0;
                        for (int y = ly; y < ey; y++)
                            for (int x = lx; x < ex; x++)
                            {
                                float b = input[(c * g.ImageSize + y) * g.ImageSize + x, n];
                                if (Pool == "avg") value += b;
                                else if (b > value) value = b;
                                else if (value == 0 && b == 0)
                                    value = BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(value) & BitConverter.SingleToInt32Bits(b));
                            }
                        target[(c * g.OutputSize + oy) * g.OutputSize + ox, n] =
                            Pool == "avg" ? value / ((ey - ly) * (ex - lx)) : value;
                    }
                }
    }

    /// <summary>For max, input and saved pre-neuron pool outputs are required.
    /// For avg these may be null. Zero scaleTargets never reads the old target;
    /// other scales multiply the target once after summing overlapping outputs.</summary>
    public void Backward(FloatMatrix? input, FloatMatrix outputGradient, FloatMatrix? output,
        FloatMatrix target, float scaleTargets = 0)
    {
        var g = Geometry;
        PoolContract.Matrix(Name, outputGradient, g.OutputRows, target.Cols);
        PoolContract.Matrix(Name, target, g.InputRows, outputGradient.Cols);
        PoolContract.Disjoint(Name, target, outputGradient);
        if (Pool == "max")
        {
            ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(output);
            PoolContract.Matrix(Name, input, g.InputRows, target.Cols);
            PoolContract.Matrix(Name, output, g.OutputRows, target.Cols);
            PoolContract.Disjoint(Name, target, input, output);
        }
        for (int c = 0; c < g.Channels; c++)
            for (int y = 0; y < g.ImageSize; y++)
                for (int x = 0; x < g.ImageSize; x++)
                    for (int n = 0; n < target.Cols; n++)
                    {
                        float sum = 0;
                        int minY = Math.Max(0, (int)Math.Ceiling((y - (double)g.Start - g.SizeX + 1) / g.Stride));
                        int minX = Math.Max(0, (int)Math.Ceiling((x - (double)g.Start - g.SizeX + 1) / g.Stride));
                        int endY = Math.Min(g.OutputSize, (int)Math.Floor((y - (double)g.Start) / g.Stride) + 1);
                        int endX = Math.Min(g.OutputSize, (int)Math.Floor((x - (double)g.Start) / g.Stride) + 1);
                        int row = (c * g.ImageSize + y) * g.ImageSize + x;
                        for (int oy = minY; oy < endY; oy++)
                            for (int ox = minX; ox < endX; ox++)
                            {
                                int o = (c * g.OutputSize + oy) * g.OutputSize + ox;
                                if (Pool == "max") sum += (input![row, n] == output![o, n] ? 1f : 0f) * outputGradient[o, n];
                                else
                                {
                                    int sy = g.Start + oy * g.Stride, sx = g.Start + ox * g.Stride;
                                    int area = (Math.Min(g.ImageSize, sy + g.SizeX) - Math.Max(0, sy)) *
                                        (Math.Min(g.ImageSize, sx + g.SizeX) - Math.Max(0, sx));
                                    sum += outputGradient[o, n] * (1f / area);
                                }
                            }
                        target[row, n] = scaleTargets == 0 ? sum : scaleTargets * target[row, n] + sum;
                    }
    }
}
