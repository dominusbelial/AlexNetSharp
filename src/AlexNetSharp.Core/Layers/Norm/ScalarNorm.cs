using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Norm;

/// <summary>Plain float32 reference. Sums ascending channels (cmrnorm) or y,x
/// (rnorm), starting at +0. Uses MathF.Pow, separate multiply/add and division.
/// A:src/cudaconv2/conv_util.cu:758-804,1068-1086,1142-1175,1593-1605,1643-1691.
/// Candidate A uses __powf/__fdividef; CPU bitwise historical CUDA parity is not claimed.</summary>
public sealed class ScalarNorm
{
    public string Name { get; }
    public NormGeometry Geometry { get; }
    public string Kind { get; }
    /// <summary>Already parser-divided scale, not the raw config value.</summary>
    public float AddScale { get; }
    public float Power { get; }
    public float MinDiv { get; }
    public bool Blocked { get; }

    public ScalarNorm(string name, NormGeometry geometry, string kind, float addScale, float power,
        float minDiv = 1, bool blocked = false)
    {
        NormContract.Validate(name, geometry, kind);
        Name = name; Geometry = geometry; Kind = kind; AddScale = addScale;
        Power = power; MinDiv = minDiv; Blocked = blocked;
    }

    public static ScalarNorm FromLayer(LayerDefinition layer, LayerShape shape)
    {
        var p = NormContract.FromLayer(layer, shape);
        return new(layer.Name, p.Geometry, layer.Type, p.Scale, p.Power, p.MinDiv, p.Blocked);
    }

    /// <summary>Always overwrites both output and denominator cache. All buffers
    /// are caller-owned. Spatial rnorm ignores MinDiv and Blocked, as the original does.</summary>
    public void Forward(FloatMatrix input, FloatMatrix denoms, FloatMatrix target)
    {
        int rows = Geometry.Rows, size = Geometry.ImageSize, half = Geometry.Size / 2;
        NormContract.Matrix(Name, input, rows, target.Cols);
        NormContract.Matrix(Name, denoms, rows, input.Cols);
        NormContract.Matrix(Name, target, rows, input.Cols);
        NormContract.Disjoint(Name, target, input, denoms);
        NormContract.Disjoint(Name, denoms, input);
        for (int c = 0; c < Geometry.Channels; c++)
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    for (int n = 0; n < input.Cols; n++)
                    {
                        float sum = 0;
                        if (Kind == "cmrnorm")
                        {
                            int start = Blocked ? c / Geometry.Size * Geometry.Size : c - half;
                            for (int f = Math.Max(0, start); f < Math.Min(Geometry.Channels, start + Geometry.Size); f++)
                            {
                                float v = input[(f * size + y) * size + x, n];
                                sum += v * v;
                            }
                        }
                        else
                            for (int yy = Math.Max(0, y - half); yy < Math.Min(size, y - half + Geometry.Size); yy++)
                                for (int xx = Math.Max(0, x - half); xx < Math.Min(size, x - half + Geometry.Size); xx++)
                                {
                                    float v = input[(c * size + yy) * size + xx, n];
                                    sum += v * v;
                                }
                        int row = (c * size + y) * size + x;
                        float denom = (Kind == "cmrnorm" ? MinDiv : 1f) + AddScale * sum;
                        denoms[row, n] = denom;
                        target[row, n] = input[row, n] * MathF.Pow(denom, -Power);
                    }
    }

    /// <summary>Uses the saved output and denominator, never recomputes them.
    /// Unlike CUDA's destructive acts preliminaries, preserves all sources with one
    /// temporary (Rows,Cases) buffer. Zero scaleTargets never reads the old target;
    /// nonzero scales multiply it once after the complete input gradient.</summary>
    public void Backward(FloatMatrix input, FloatMatrix outputGradient, FloatMatrix denoms,
        FloatMatrix output, FloatMatrix target, float scaleTargets = 0)
    {
        int rows = Geometry.Rows, size = Geometry.ImageSize, half = Geometry.Size / 2;
        foreach (var m in new[] { input, outputGradient, denoms, output, target })
            NormContract.Matrix(Name, m, rows, input.Cols);
        NormContract.Disjoint(Name, target, input, outputGradient, denoms, output);
        var tmp = new FloatMatrix(rows, input.Cols);
        float coefficient = -2f * AddScale * Power;
        for (int r = 0; r < rows; r++)
            for (int n = 0; n < input.Cols; n++)
                tmp[r, n] = coefficient * outputGradient[r, n] * output[r, n] / denoms[r, n];
        for (int c = 0; c < Geometry.Channels; c++)
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    for (int n = 0; n < input.Cols; n++)
                    {
                        float sum = 0;
                        if (Kind == "cmrnorm")
                        {
                            int start = Blocked ? c / Geometry.Size * Geometry.Size : c + half - Geometry.Size + 1;
                            for (int f = Math.Max(0, start); f < Math.Min(Geometry.Channels, start + Geometry.Size); f++)
                                sum += tmp[(f * size + y) * size + x, n];
                        }
                        else
                            for (int yy = Math.Max(0, y + half - Geometry.Size + 1); yy < Math.Min(size, y + half + 1); yy++)
                                for (int xx = Math.Max(0, x + half - Geometry.Size + 1); xx < Math.Min(size, x + half + 1); xx++)
                                    sum += tmp[(c * size + yy) * size + xx, n];
                        int row = (c * size + y) * size + x;
                        float value = input[row, n] * sum + outputGradient[row, n] * MathF.Pow(denoms[row, n], -Power);
                        target[row, n] = scaleTargets == 0 ? value : scaleTargets * target[row, n] + value;
                    }
    }
}
