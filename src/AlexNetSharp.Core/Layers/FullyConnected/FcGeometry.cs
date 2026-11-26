using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.FullyConnected;

/// <summary>Native FC matrices: X=(cases,width) F, W=(width,outputs) F,
/// Y=(cases,outputs) F, b=(1,outputs) F. Canonical activations use TransposeView().
/// Original layer.py:883-884; src/layer.cu:152-165,635.</summary>
public sealed class FcGeometry
{
    public string Name { get; }
    public IReadOnlyList<int> InputWidths { get; }
    public int Outputs { get; }

    public FcGeometry(string name, IReadOnlyList<int> inputWidths, int outputs)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(inputWidths);
        if (inputWidths.Count == 0 || inputWidths.Any(k => k <= 0) || outputs <= 0)
            throw new ArgumentException($"Layer '{name}': FC dimensions must be positive with at least one input.");
        foreach (int k in inputWidths) _ = checked(k * outputs);
        Name = name; InputWidths = Array.AsReadOnly(inputWidths.ToArray()); Outputs = outputs;
    }

    public static FcGeometry FromLayer(LayerDefinition layer, LayerShape shape)
    {
        if (layer.TypedParams is not FcLayerParams p || shape.Type != "fc" || layer.Name != shape.Name)
            throw new ArgumentException("Matching FC definition and shape required.");
        if (p.WeightSource.Any(s => s.Length != 0) || p.WeightSourceLayers.Any(s => s.Length != 0))
            throw new NotSupportedException($"Layer '{layer.Name}': weightSource / weight sharing is unsupported in this port.");
        if (!layer.Inputs.SequenceEqual(shape.Inputs) || shape.NumInputs.Count != layer.Inputs.Count ||
            shape.Outputs != p.Outputs || shape.WeightsShapes.Count != layer.Inputs.Count ||
            !shape.BiasesShape.SequenceEqual(new[] { 1, p.Outputs }) ||
            shape.WeightsShapes.Where((w, i) => !w.SequenceEqual(new[] { shape.NumInputs[i], p.Outputs })).Any())
            throw new ArgumentException($"Layer '{layer.Name}': definition and shape disagree.");
        return new(layer.Name, shape.NumInputs, p.Outputs);
    }

    internal int Width(int input)
    {
        if ((uint)input >= (uint)InputWidths.Count) throw new ArgumentOutOfRangeException(nameof(input));
        return InputWidths[input];
    }
    internal void Matrix(FloatMatrix m, int rows, int cols)
    {
        if (m.Layout != MatrixLayout.ColumnMajor || m.RowStride != 1 || m.Rows != rows || m.Cols != cols || rows <= 0)
            throw new ArgumentException($"Layer '{Name}': expected native F-order ({rows},{cols}) matrix.");
    }
    internal void Disjoint(FloatMatrix target, params FloatMatrix[] sources)
    {
        if (sources.Any(s => ReferenceEquals(target.BackingArray, s.BackingArray)))
            throw new ArgumentException($"Layer '{Name}': FC target must not share operand storage.");
    }
    internal void Forward(int input, FloatMatrix x, FloatMatrix w, FloatMatrix b, FloatMatrix y)
    {
        int k = Width(input);
        Matrix(y, y.Rows, Outputs); Matrix(x, y.Rows, k); Matrix(w, k, Outputs); Matrix(b, 1, Outputs);
        Disjoint(y, x, w, b);
    }
    internal void ForwardAll(IReadOnlyList<FloatMatrix> x, IReadOnlyList<FloatMatrix> w, FloatMatrix b, FloatMatrix y)
    {
        if (x.Count != InputWidths.Count || w.Count != InputWidths.Count)
            throw new ArgumentException($"Layer '{Name}': one X and W per declared input required.");
        for (int i = 0; i < x.Count; i++) Forward(i, x[i], w[i], b, y);
    }
    internal void Input(int input, FloatMatrix dy, FloatMatrix w, FloatMatrix dx)
    {
        int k = Width(input);
        Matrix(dy, dy.Rows, Outputs); Matrix(w, k, Outputs); Matrix(dx, dy.Rows, k); Disjoint(dx, dy, w);
    }
    internal void Weight(int input, FloatMatrix x, FloatMatrix dy, FloatMatrix dw)
    {
        int k = Width(input);
        Matrix(dy, dy.Rows, Outputs); Matrix(x, dy.Rows, k); Matrix(dw, k, Outputs); Disjoint(dw, x, dy);
    }
    internal void Bias(FloatMatrix dy, FloatMatrix db)
    {
        Matrix(dy, dy.Rows, Outputs); Matrix(db, 1, Outputs); Disjoint(db, dy);
    }
}
