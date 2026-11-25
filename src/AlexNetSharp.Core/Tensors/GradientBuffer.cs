namespace AlexNetSharp.Core.Tensors;

/// <summary>A single accumulator (one device in the original). First contribution overwrites,
/// subsequent contributions accumulate. BeginPass resets the count without reading/clearing
/// old values (src/layer.cu:287-295,329-334). Not thread safe.</summary>
public sealed class GradientBuffer
{
    public FloatMatrix Matrix { get; }
    public int Contributions { get; private set; }
    public float NextScaleTargets => Contributions == 0 ? 0 : 1;

    public GradientBuffer(FloatMatrix matrix) => Matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));

    public void BeginPass() => Contributions = 0;

    /// <returns>The scaleTargets used for this contribution.</returns>
    public float Add(FloatMatrix contribution, float scaleSource = 1)
    {
        float scale = NextScaleTargets;
        Matrix.Add(contribution, scale, scaleSource);
        Contributions++;
        return scale;
    }

    public float AddProduct(FloatMatrix a, FloatMatrix b, float scaleAB = 1)
    {
        float scale = NextScaleTargets;
        Matrix.AddProduct(a, b, scale, scaleAB);
        Contributions++;
        return scale;
    }
}
