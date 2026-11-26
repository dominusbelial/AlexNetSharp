using AlexNetSharp.Core.Layers.FullyConnected;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal sealed class FullyConnectedAdapter
{
    private readonly ScalarFullyConnected? _scalar;
    private readonly OptimizedFullyConnected? _fast;
    internal FullyConnectedAdapter(bool fast, FcGeometry geometry, int threads = -1)
    {
        if (fast) _fast = new(geometry, threads); else _scalar = new(geometry);
    }
    internal void Forward(IReadOnlyList<FloatMatrix> x, IReadOnlyList<FloatMatrix> w, FloatMatrix b, FloatMatrix y)
    { if (_fast != null) _fast.Forward(x, w, b, y); else _scalar!.Forward(x, w, b, y); }
    internal void ForwardInput(int i, FloatMatrix x, FloatMatrix w, FloatMatrix b, FloatMatrix y, float beta = 0)
    { if (_fast != null) _fast.ForwardInput(i, x, w, b, y, beta); else _scalar!.ForwardInput(i, x, w, b, y, beta); }
    internal void Input(int i, FloatMatrix dy, FloatMatrix w, FloatMatrix dx, float beta = 0)
    { if (_fast != null) _fast.BackwardInput(i, dy, w, dx, beta); else _scalar!.BackwardInput(i, dy, w, dx, beta); }
    internal void Weight(int i, FloatMatrix x, FloatMatrix dy, FloatMatrix dw, float alpha = 1, float beta = 0)
    { if (_fast != null) _fast.BackwardWeights(i, x, dy, dw, alpha, beta); else _scalar!.BackwardWeights(i, x, dy, dw, alpha, beta); }
    internal void Bias(FloatMatrix dy, FloatMatrix db, float alpha = 1, float beta = 0)
    { if (_fast != null) _fast.BackwardBiases(dy, db, alpha, beta); else _scalar!.BackwardBiases(dy, db, alpha, beta); }
}
