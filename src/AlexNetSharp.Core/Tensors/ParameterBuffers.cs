using AlexNetSharp.Core.Shapes;

namespace AlexNetSharp.Core.Tensors;

public abstract class WeightBuffer(string inputName, FloatMatrix matrix)
{
    public string InputName { get; } = inputName;
    public FloatMatrix Matrix { get; } = matrix;
}

/// <summary>C-order (filterChannels*filterPixels,filters); layer.py:1072.
/// Filter rows are channel,y,x; filter index is fastest in storage.</summary>
public sealed class ConvWeightBuffer : WeightBuffer
{
    public ActivationLayout FilterLayout { get; }
    public ConvWeightBuffer(string inputName, int channels, int height, int width, int filters)
        : this(inputName, new ActivationLayout(channels, height, width, filters)) { }

    private ConvWeightBuffer(string inputName, ActivationLayout layout)
        : base(inputName, new FloatMatrix(layout.Rows, layout.Cases)) => FilterLayout = layout;

    public float this[int channel, int y, int x, int filter]
    {
        get => Matrix[FilterLayout.GetRow(channel, y, x), filter];
        set => Matrix[FilterLayout.GetRow(channel, y, x), filter] = value;
    }
}

/// <summary>F-order (numInputs,outputs), one matrix per input (layer.py:883).</summary>
public sealed class FcWeightBuffer(string inputName, int inputs, int outputs)
    : WeightBuffer(inputName, new FloatMatrix(inputs, outputs, MatrixLayout.ColumnMajor));

public sealed class LayerParameterBuffers
{
    public IReadOnlyList<WeightBuffer> Weights { get; }
    public FloatMatrix? Biases { get; }
    public long ParameterCount => Weights.Sum(w => (long)w.Matrix.ElementCount) + (Biases?.ElementCount ?? 0);

    private LayerParameterBuffers(WeightBuffer[] weights, FloatMatrix? biases)
    {
        Weights = Array.AsReadOnly(weights);
        Biases = biases;
    }

    /// <summary>Allocate zeroed parameter slots from ANX-3 shapes, retaining input order
    /// (layer.py:797-807). No initialization, sharing, gradients or optimizer state.</summary>
    public static LayerParameterBuffers Allocate(LayerShape shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var weights = new WeightBuffer[shape.WeightsShapes.Count];
        for (int i = 0; i < weights.Length; i++)
        {
            var dims = shape.WeightsShapes[i];
            if (shape.Type == "conv")
            {
                int side = (int)Math.Sqrt(shape.FilterPixels[i]);
                if (side * side != shape.FilterPixels[i])
                    throw new ArgumentException("Convolution filter must be square.", nameof(shape));
                weights[i] = new ConvWeightBuffer(shape.Inputs[i], shape.FilterChannels[i], side, side, dims[1]);
            }
            else if (shape.Type == "fc") weights[i] = new FcWeightBuffer(shape.Inputs[i], dims[0], dims[1]);
            else throw new ArgumentException("Weight buffers require a conv or fc shape.", nameof(shape));
            if (weights[i].Matrix.Rows != dims[0] || weights[i].Matrix.Cols != dims[1])
                throw new ArgumentException("Weight shape and filter dimensions disagree.", nameof(shape));
        }
        FloatMatrix? biases = shape.BiasesShape.Count == 0 ? null : new(shape.BiasesShape[0], shape.BiasesShape[1],
            shape.Type == "fc" ? MatrixLayout.ColumnMajor : MatrixLayout.RowMajor);
        return new(weights, biases);
    }
}

/// <summary>One logical layer's independent buffers. The original may share activations
/// in place; this scalar foundation allocates independently and does not implement a GPU runtime.</summary>
public sealed class LayerBuffers
{
    public FloatMatrix? Activations { get; }
    public LayerParameterBuffers Parameters { get; }

    private LayerBuffers(FloatMatrix? activations, LayerParameterBuffers parameters)
    {
        Activations = activations;
        Parameters = parameters;
    }

    public static bool UsesTransposedActivations(string type) => type switch
    {
        "fc" or "softmax" or "neuron" => true, // src/layer.cu:484,635,1017
        "data" or "conv" or "pool" or "cmrnorm" or "rnorm" or "hs" or "cost.logreg" => false,
        _ => throw new ArgumentException("Unsupported layer type.", nameof(type)),
    };

    public static LayerBuffers Allocate(LayerShape shape, int cases)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cases);
        bool trans = UsesTransposedActivations(shape.Type);
        // Cost has outputs=0 and no activation output (layer.py:187,1261-1273).
        var canonical = shape.Outputs == 0 ? null : new FloatMatrix(shape.Outputs, cases);
        return new(trans ? canonical?.TransposeView() : canonical, LayerParameterBuffers.Allocate(shape));
    }
}
