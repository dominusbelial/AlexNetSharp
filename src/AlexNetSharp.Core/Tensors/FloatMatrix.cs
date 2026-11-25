namespace AlexNetSharp.Core.Tensors;

public enum MatrixLayout { RowMajor, ColumnMajor }

/// <summary>
/// Float32 matrix with explicit logical dimensions and physical strides. ColumnMajor is
/// NVMatrix.isTrans=true (Candidate A include/nvmatrix/nvmatrix.cuh:120-160).
/// Views share storage; transposing a view swaps dimensions/strides without moving data
/// (Candidate A src/nvmatrix/nvmatrix.cu:530-550). No operation silently changes layout.
/// </summary>
public sealed class FloatMatrix
{
    private readonly float[] _storage;
    public int Rows { get; }
    public int Cols { get; }
    public MatrixLayout Layout { get; }
    public bool IsTrans => Layout == MatrixLayout.ColumnMajor;
    public int LeadingDimension => IsTrans ? Rows : Cols;
    public int FollowingDimension => IsTrans ? Cols : Rows;
    public int RowStride { get; }
    public int ColumnStride { get; }
    public int StorageOffset { get; }
    public int ElementCount => checked(Rows * Cols);
    public bool IsContiguous => ElementCount == 0 ||
        (Rows <= 1 || RowStride == (IsTrans ? 1 : Cols)) &&
        (Cols <= 1 || ColumnStride == (IsTrans ? Rows : 1));

    /// <summary>Entire backing allocation, including elements outside a view.</summary>
    public Memory<float> Storage => _storage;
    internal float[] BackingArray => _storage;

    public FloatMatrix(int rows, int cols, MatrixLayout layout = MatrixLayout.RowMajor)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(cols);
        if (!Enum.IsDefined(layout)) throw new ArgumentOutOfRangeException(nameof(layout));
        Rows = rows;
        Cols = cols;
        Layout = layout;
        RowStride = IsTrans ? 1 : cols;
        ColumnStride = IsTrans ? rows : 1;
        _storage = new float[checked(rows * cols)];
    }

    internal FloatMatrix(float[] storage, int offset, int rows, int cols,
        int rowStride, int columnStride, MatrixLayout layout)
    {
        _storage = storage;
        StorageOffset = offset;
        Rows = rows;
        Cols = cols;
        RowStride = rowStride;
        ColumnStride = columnStride;
        Layout = layout;
    }

    /// <summary>Absolute element offset within Storage, including view offset.</summary>
    public int GetFlatOffset(int row, int col)
    {
        CheckIndex(row, Rows, nameof(row));
        CheckIndex(col, Cols, nameof(col));
        return checked(StorageOffset + row * RowStride + col * ColumnStride);
    }

    public float this[int row, int col]
    {
        get => _storage[GetFlatOffset(row, col)];
        set => _storage[GetFlatOffset(row, col)] = value;
    }

    public FloatMatrix TransposeView() => new(_storage, StorageOffset, Cols, Rows,
        ColumnStride, RowStride, IsTrans ? MatrixLayout.RowMajor : MatrixLayout.ColumnMajor);

    public FloatMatrix Slice(int firstRow, int rows, int firstCol, int cols)
    {
        CheckRange(firstRow, rows, Rows, nameof(firstRow));
        CheckRange(firstCol, cols, Cols, nameof(firstCol));
        int offset = rows == 0 || cols == 0 ? StorageOffset : GetFlatOffset(firstRow, firstCol);
        return new(_storage, offset, rows, cols, RowStride, ColumnStride, Layout);
    }

    /// <summary>Layer::getNumCases, including virtual batch size for deferred updates
    /// (src/layer.cu:337-340). layerTrans is the layer's _trans flag.</summary>
    public int GetNumCases(bool layerTrans, int weightUpdateFrequency = 1, int minibatchSize = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(weightUpdateFrequency);
        if (weightUpdateFrequency == 1) return layerTrans ? Rows : Cols;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minibatchSize);
        return checked(weightUpdateFrequency * minibatchSize);
    }

    public void Fill(float value)
    {
        for (int row = 0; row < Rows; row++)
            for (int col = 0; col < Cols; col++) this[row, col] = value;
    }

    /// <summary>this = scaleTargets*this + scaleSource*source. Zero scaleTargets
    /// does not read old target values, even NaN (Candidate A nvmatrix.cu:583-592).
    /// Supports either explicit layout; partial overlapping views are rejected.</summary>
    public void Add(FloatMatrix source, float scaleTargets = 1, float scaleSource = 1)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (Rows != source.Rows || Cols != source.Cols)
            throw new ArgumentException("Addition requires equal logical dimensions.", nameof(source));
        if (ReferenceEquals(_storage, source._storage) &&
            (StorageOffset != source.StorageOffset || RowStride != source.RowStride || ColumnStride != source.ColumnStride))
            throw new ArgumentException("Addition does not support overlapping views with different index maps.", nameof(source));
        for (int row = 0; row < Rows; row++)
            for (int col = 0; col < Cols; col++)
                this[row, col] = scaleTargets == 0 ? scaleSource * source[row, col]
                    : scaleTargets * this[row, col] + scaleSource * source[row, col];
    }

    /// <summary>this = scaleThis*this + scaleAB*(a*b), using scalar float loops.
    /// Zero scaleThis skips the old target (Candidate A nvmatrix.cu:221-236,190-205,
    /// SGEMM beta=0). As with BLAS, alpha=0 skips the product operands.
    /// Shapes/layouts are preallocated and remain explicit; either output layout is supported.
    /// Product target must have separate backing storage from both operands.</summary>
    public void AddProduct(FloatMatrix a, FloatMatrix b, float scaleThis = 1, float scaleAB = 1)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Cols != b.Rows || Rows != a.Rows || Cols != b.Cols)
            throw new ArgumentException("Product dimensions must be (m,k)*(k,n) with an (m,n) target.");
        if (ReferenceEquals(_storage, a._storage) || ReferenceEquals(_storage, b._storage))
            throw new ArgumentException("Product target must not share backing storage with its operands.");
        for (int row = 0; row < Rows; row++)
            for (int col = 0; col < Cols; col++)
            {
                float product = 0;
                if (scaleAB != 0)
                {
                    for (int k = 0; k < a.Cols; k++) product += a[row, k] * b[k, col];
                    product *= scaleAB;
                }
                this[row, col] = scaleThis == 0 ? product : scaleThis * this[row, col] + product;
            }
    }

    internal static void CheckIndex(int index, int length, string name)
    {
        if ((uint)index >= (uint)length) throw new ArgumentOutOfRangeException(name);
    }

    private static void CheckRange(int start, int count, int length, string name)
    {
        if (start < 0 || count < 0 || start > length || count > length - start)
            throw new ArgumentOutOfRangeException(name);
    }
}
