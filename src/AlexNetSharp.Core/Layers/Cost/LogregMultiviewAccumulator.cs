using AlexNetSharp.Core.Layers.Softmax;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Cost;

[Flags]
public enum MultiviewPhase { Continue = 0, Start = 1, End = 2 }

/// <summary>Only cost-owned state from src/layer.cu:1804-1836. Start copies;
/// continue adds probabilities; end scales the saved sum by float32(1.0/count)
/// and computes cost. Intermediate calls return null and retain LastCost.
/// Start|End supports one view. Start resets even an incomplete sequence.
/// Calls are sequential, with parallel cases inside optimized accumulation.
/// A fresh start is required after end; invalid sequences/shapes throw.</summary>
public sealed class LogregMultiviewAccumulator(bool optimized = false, int maxDegreeOfParallelism = -1)
{
    private FloatMatrix? _probabilities;
    private bool _active;
    public int ViewCount { get; private set; }
    public LogregCostResult? LastCost { get; private set; }
    public FloatMatrix CopyProbabilities()
    {
        if (_probabilities == null) throw new InvalidOperationException("No accumulated probabilities.");
        var copy = new FloatMatrix(_probabilities.Rows, _probabilities.Cols); copy.Add(_probabilities, 0); return copy;
    }
    public LogregCostResult? Forward(FloatMatrix labels, FloatMatrix probabilities, int topK, MultiviewPhase phase)
    {
        LogregContract.Validate(labels, probabilities, topK); SoftmaxContract.Threads(maxDegreeOfParallelism);
        if ((phase & ~(MultiviewPhase.Start | MultiviewPhase.End)) != 0) throw new ArgumentOutOfRangeException(nameof(phase));
        bool start = phase.HasFlag(MultiviewPhase.Start), end = phase.HasFlag(MultiviewPhase.End);
        if (!start && (!_active || _probabilities!.Rows != probabilities.Rows || _probabilities.Cols != probabilities.Cols))
            throw new InvalidOperationException("Multiview requires start and matching dimensions.");
        if (start) { _probabilities = new(probabilities.Rows, probabilities.Cols); ViewCount = 0; _active = true; }
        ViewCount++;
        float scale = (float)(1.0 / ViewCount);
        if (optimized)
        {
            var x = probabilities.BackingArray; var a = _probabilities!.BackingArray;
            Parallel.For(0, probabilities.Cols, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, n =>
            {
                for (int c = 0; c < probabilities.Rows; c++)
                {
                    int i = c * probabilities.Cols + n;
                    float v = x[probabilities.StorageOffset + c * probabilities.RowStride + n * probabilities.ColumnStride];
                    a[i] = start ? v : a[i] + v;
                    if (end) a[i] *= scale;
                }
            });
        }
        else
        {
            for (int c = 0; c < probabilities.Rows; c++) for (int n = 0; n < probabilities.Cols; n++)
            {
                _probabilities![c, n] = start ? probabilities[c, n] : _probabilities[c, n] + probabilities[c, n];
                if (end) _probabilities[c, n] *= scale;
            }
        }
        if (!end) return null;
        _active = false;
        LastCost = optimized ? OptimizedLogregCost.Forward(labels, _probabilities!, topK, maxDegreeOfParallelism) : ScalarLogregCost.Forward(labels, _probabilities!, topK);
        return LastCost;
    }
}
