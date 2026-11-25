using System.Text.Json.Nodes;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class TensorTests
{
    public static int RunAll(string repoRoot, string origRoot)
    {
        int count = 0;
        void Run(Action test) { test(); count++; }
        Run(Index13);
        Run(Index55);
        Run(MatrixViews);
        Run(ActivationViews);
        Run(NchwHandFixture);
        foreach (var dims in new[] { (3, 5, 7, 9), (1, 3, 13, 13), (5, 1, 1, 3), (3, 1, 1, 1) })
            Run(() => RoundTrip(dims.Item1, dims.Item2, dims.Item3, dims.Item4));
        Run(RgbCrop);
        Run(ConvWeightOrder);
        Run(FcWeightOrder);
        foreach (var a in Enum.GetValues<MatrixLayout>())
            foreach (var b in Enum.GetValues<MatrixLayout>())
                foreach (var target in Enum.GetValues<MatrixLayout>())
                    Run(() => Product(a, b, target));
        Run(ProductViews);
        Run(ProductZeroScales);
        Run(AddScales);
        Run(FanOut);
        Run(FanOutProducts);
        Run(RemoteOnlyMerge);
        Run(Validation);
        var net = ShapeTests.Parse145(origRoot);
        Run(() => GraphFanOut(net));
        var fixture = JsonNode.Parse(File.ReadAllText(Path.Combine(repoRoot, "tests", "fixtures", "layers-145.shapes.json")))!;
        foreach (var c in fixture["cases"]!.AsArray())
            Run(() => Allocate145(net, c!));
        return count;
    }

    private static void Index13()
    {
        var layout = new ActivationLayout(3, 13, 13, 5);
        // c,y,x,n,row,offset: independent hand arithmetic, including all corners.
        int[][] cases = [[0,0,0,0,0,0], [0,0,12,4,12,64], [0,12,0,4,156,784],
            [0,12,12,4,168,844], [2,0,0,0,338,1690], [2,12,12,4,506,2534]];
        CheckIndices(layout, cases);
        TestAssert.Equal(2535, layout.ElementCount, "13 plane element count");
    }

    private static void Index55()
    {
        var layout = new ActivationLayout(48, 55, 55, 3);
        int[][] cases = [[0,0,0,0,0,0], [0,0,54,2,54,164], [0,54,0,2,2970,8912],
            [0,54,54,2,3024,9074], [47,0,0,0,142175,426525], [47,54,54,2,145199,435599]];
        CheckIndices(layout, cases);
        TestAssert.Equal(435600, layout.ElementCount, "55 plane element count");
    }

    private static void CheckIndices(ActivationLayout layout, int[][] fixtures)
    {
        var matrix = new FloatMatrix(layout.Rows, layout.Cases);
        foreach (var f in fixtures)
        {
            var index = new ActivationIndex(f[0], f[1], f[2], f[3]);
            TestAssert.Equal((f[4], f[3]), layout.ToMatrixIndex(f[0], f[1], f[2], f[3]), "hand matrix index");
            TestAssert.Equal(f[5], layout.GetFlatOffset(f[0], f[1], f[2], f[3]), "hand flat offset");
            TestAssert.Equal(f[5], matrix.GetFlatOffset(f[4], f[3]), "matrix physical offset");
            TestAssert.Equal(index, layout.FromMatrixIndex(f[4], f[3]), "inverse row/col");
            TestAssert.Equal(index, layout.FromFlatOffset(f[5]), "inverse flat");
        }
    }

    private static void MatrixViews()
    {
        var m = Matrix(2, 3, MatrixLayout.RowMajor, [1,2,3,4,5,6]);
        TestAssert.SequenceEqual<float>([1,2,3,4,5,6], m.Storage.ToArray(), "C order");
        TestAssert.Equal(3, m.LeadingDimension, "row-major leading dimension");
        TestAssert.Equal(2, m.FollowingDimension, "row-major following dimension");
        var t = m.TransposeView();
        TestAssert.Equal((3, 2, true), (t.Rows, t.Cols, t.IsTrans), "transposed logical dimensions");
        TestAssert.Equal((3, 2), (t.LeadingDimension, t.FollowingDimension), "transposed dimensions follow physical storage");
        TestAssert.Equal(5f, t[1,1], "transposed coordinate");
        t[2,1] = 99;
        TestAssert.Equal(99f, m[1,2], "transpose shares storage");
        CheckMatrix(t.TransposeView(), [1,2,3,4,5,99]);
        TestAssert.True(m.IsContiguous && t.IsContiguous, "both complete views contiguous");
        var slice = m.Slice(0, 2, 1, 2);
        TestAssert.Equal((1, 3, 1), (slice.StorageOffset, slice.RowStride, slice.ColumnStride), "slice strides");
        TestAssert.Equal(4, slice.GetFlatOffset(1,0), "slice absolute offset");
        TestAssert.True(!slice.IsContiguous, "partial row slice is strided");
        slice.Fill(-7);
        CheckMatrix(m, [1,-7,-7,4,-7,-7]);
        var f = Matrix(3, 4, MatrixLayout.ColumnMajor, [1,2,3,4,5,6,7,8,9,10,11,12]);
        var fs = f.Slice(1, 2, 1, 2);
        TestAssert.Equal((4, 1, 3), (fs.StorageOffset, fs.RowStride, fs.ColumnStride), "F slice strides");
        CheckMatrix(fs.TransposeView(), [6,10,7,11]);
        TestAssert.Equal(3, m.GetNumCases(false), "explicit untransposed getNumCases cols");
    }

    private static void ActivationViews()
    {
        var t = new ActivationTensor(3, 3, 5, 3);
        t.Matrix.Fill(-1);
        var oneCase = t.CaseView(2);
        TestAssert.Equal((45, 1, 3), (oneCase.Rows, oneCase.Cols, oneCase.RowStride), "one case view");
        oneCase[44,0] = 17;
        TestAssert.Equal(17f, t[2,2,4,2], "case view write");
        var plane = t.ChannelPlaneView(2, 2);
        TestAssert.Equal((3, 5, 15, 3, 92), (plane.Rows, plane.Cols, plane.RowStride, plane.ColumnStride, plane.StorageOffset), "plane view map");
        plane[0,4] = 31;
        t[2,1,2,2] = 47;
        TestAssert.Equal(47f, plane[1,2], "plane sees tensor writes");
        TestAssert.Equal(31f, oneCase[34,0], "case sees plane writes");
        TestAssert.Equal(-1f, t[2,0,4,1], "other case untouched");
        TestAssert.Equal(-1f, t[1,0,4,2], "other channel untouched");
        var trans = t.Matrix.TransposeView();
        TestAssert.Equal(3, t.Matrix.GetNumCases(false), "untransposed cases");
        TestAssert.Equal(3, trans.GetNumCases(true), "transposed cases");
        TestAssert.Equal(384, trans.GetNumCases(true, 3, 128), "deferred update virtual batch");
        TestAssert.Equal(31f, trans[2,34], "transposed case-first logical view");
    }

    private static void NchwHandFixture()
    {
        // Two cases, two channels, 1x3 planes. Explicit expected interleaving.
        float[] nchw = [1,2,3,11,12,13,101,102,103,111,112,113];
        var t = LayoutBridge.FromNchw(nchw, 2, 2, 1, 3);
        TestAssert.SequenceEqual<float>([1,101,2,102,3,103,11,111,12,112,13,113], t.Matrix.Storage.ToArray(), "hand original bytes");
        TestAssert.SequenceEqual(nchw, LayoutBridge.ToNchw(t), "hand inverse");
        nchw[0] = -99;
        TestAssert.Equal(1f, t[0,0,0,0], "import copies source");
    }

    private static void RoundTrip(int n, int c, int h, int w)
    {
        var source = new float[n * c * h * w];
        for (int i = 0; i < source.Length; i++) source[i] = i - 50.25f;
        // Include special float32 bit patterns; conversion must not do arithmetic.
        source[0] = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
        source[^1] = BitConverter.Int32BitsToSingle(0x7fc01234);
        var tensor = LayoutBridge.FromNchw(source, n, c, h, w);
        for (int ci = 0; ci < c; ci++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int ni = 0; ni < n; ni++)
                    {
                        int referenceOffset = ((ni * c + ci) * h + y) * w + x;
                        TestAssert.Equal(BitConverter.SingleToInt32Bits(source[referenceOffset]),
                            BitConverter.SingleToInt32Bits(tensor[ci,y,x,ni]), "independent NCHW coordinate");
                    }
        TestAssert.SequenceEqual(source.Select(BitConverter.SingleToInt32Bits).ToArray(),
            LayoutBridge.ToNchw(tensor).Select(BitConverter.SingleToInt32Bits).ToArray(), "bit-preserving round trip");
    }

    private static void RgbCrop()
    {
        var crop = new byte[224 * 224 * 3];
        // Distinct RGB and coordinate patterns catch channel swaps, x/y swaps and /255.
        for (int y = 0; y < 224; y++)
            for (int x = 0; x < 224; x++)
            {
                int i = (y * 224 + x) * 3;
                crop[i] = (byte)x;
                crop[i+1] = (byte)y;
                crop[i+2] = (byte)((x + 2*y + 17) % 256);
            }
        var t = new ActivationTensor(3, 224, 224, 3);
        t.Matrix.Fill(-999);
        LayoutBridge.ImportRgbCrop(crop, t, 2);
        TestAssert.Equal(223f, t[0,0,223,2], "red top right");
        TestAssert.Equal(223f, t[1,223,0,2], "green bottom left");
        TestAssert.Equal(174f, t[2,223,223,2], "blue bottom right");
        TestAssert.Equal(17f, t[2,0,0,2], "blue first pixel");
        TestAssert.Equal(31f, t[0,7,31,2], "red asymmetric pixel");
        var mean = new float[crop.Length];
        for (int i = 0; i < mean.Length; i++) mean[i] = i / (224*224) + 0.5f;
        LayoutBridge.ImportRgbCrop(crop, t, 1, mean);
        TestAssert.Equal(30.5f, t[0,7,31,1], "planar red mean");
        TestAssert.Equal(5.5f, t[1,7,31,1], "planar green mean");
        TestAssert.Equal(59.5f, t[2,7,31,1], "planar blue mean");
        for (int row = 0; row < t.Matrix.Rows; row++)
            TestAssert.Equal(-999f, t.Matrix[row,0], "untouched crop case");
    }

    private static void ConvWeightOrder()
    {
        var w = new ConvWeightBuffer("image", 2, 3, 3, 2);
        // Hand rows: c0: (00,01,02,10,11,12,20,21,22), then c1 in the same order.
        int[][] coordinates = [[0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2],
            [0,2,0], [0,2,1], [0,2,2], [1,0,0], [1,0,1], [1,0,2], [1,1,0], [1,1,1],
            [1,1,2], [1,2,0], [1,2,1], [1,2,2]];
        for (int row = 0; row < coordinates.Length; row++)
        {
            var p = coordinates[row];
            TestAssert.Equal(row, w.FilterLayout.GetRow(p[0],p[1],p[2]), "hand filter row");
            w[p[0],p[1],p[2],0] = row + 1;
            w[p[0],p[1],p[2],1] = row + 101;
        }
        TestAssert.SequenceEqual<float>([1,101,2,102,3,103,4,104,5,105,6,106,7,107,8,108,9,109,
            10,110,11,111,12,112,13,113,14,114,15,115,16,116,17,117,18,118], w.Matrix.Storage.ToArray(), "hand filter C order");
    }

    private static void FcWeightOrder()
    {
        var w = new FcWeightBuffer("input", 2, 3);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 3; c++) w.Matrix[r,c] = 1 + 3*r + c;
        TestAssert.SequenceEqual<float>([1,4,2,5,3,6], w.Matrix.Storage.ToArray(), "FC F order");
        TestAssert.Equal((2,3,true), (w.Matrix.Rows,w.Matrix.Cols,w.Matrix.IsTrans), "FC logical shape retained");
    }

    private static void Product(MatrixLayout aLayout, MatrixLayout bLayout, MatrixLayout targetLayout)
    {
        var a = Matrix(2, 3, aLayout, [1,-2,3,4,0,-1]);
        var b = Matrix(3, 2, bLayout, [2,1,3,-2,5,4]);
        var target = Matrix(2, 2, targetLayout, [10,-4,6,8]);
        target.AddProduct(a, b, 0.5f, 2);
        CheckMatrix(target, [27,32,9,4]); // AB = [[11,17],[3,0]]
        target.AddProduct(a, b, 0, 1);
        CheckMatrix(target, [11,17,3,0]);
        TestAssert.Equal(targetLayout, target.Layout, "product never relayouts target");
        CheckMatrix(a, [1,-2,3,4,0,-1]);
        CheckMatrix(b, [2,1,3,-2,5,4]);
    }

    private static void ProductViews()
    {
        var aBacking = Matrix(3, 2, MatrixLayout.RowMajor, [1,4,-2,0,3,-1]);
        var a = aBacking.TransposeView();
        var bBacking = Matrix(2, 3, MatrixLayout.ColumnMajor, [2,3,5,1,-2,4]);
        var b = bBacking.TransposeView();
        var parent = new FloatMatrix(4, 6);
        parent.Fill(-99);
        var target = parent.Slice(1,2,2,2);
        target.AddProduct(a,b,0);
        CheckMatrix(target, [11,17,3,0]);
        TestAssert.Equal(-99f, parent[0,2], "product guard above");
        TestAssert.Equal(-99f, parent[1,1], "product guard left");
        TestAssert.Equal(-99f, parent[2,4], "product guard right");
        TestAssert.Equal(-99f, parent[3,2], "product guard below");
    }

    private static void ProductZeroScales()
    {
        var a = Matrix(1,2,MatrixLayout.RowMajor,[2,3]);
        var b = Matrix(2,1,MatrixLayout.ColumnMajor,[4,5]);
        var target = Matrix(1,1,MatrixLayout.ColumnMajor,[float.NaN]);
        target.AddProduct(a,b,0,2);
        CheckMatrix(target,[46]); // beta=0 must not evaluate 0*NaN
        target[0,0] = float.NaN;
        target.AddProduct(a,b,1);
        TestAssert.True(float.IsNaN(target[0,0]), "nonzero beta preserves target NaN");
        a.Fill(float.NaN);
        b.Fill(float.NaN);
        target.AddProduct(a,b,-0.0f,0);
        CheckMatrix(target,[0]); // BLAS alpha=beta=0 discards both operands and target
        target[0,0] = 8;
        target.AddProduct(a,b,-2,0);
        CheckMatrix(target,[-16]);
    }

    private static void AddScales()
    {
        var source = Matrix(2,2,MatrixLayout.ColumnMajor,[2,-4,6,8]);
        var t = Matrix(2,2,MatrixLayout.RowMajor,[float.NaN,float.NaN,float.NaN,float.NaN]);
        t.Add(source,0,0.5f);
        CheckMatrix(t,[1,-2,3,4]);
        t.Add(source,1);
        CheckMatrix(t,[3,-6,9,12]);
        t.Add(source,-2,3);
        CheckMatrix(t,[0,0,0,0]);
        t.Add(t,1);
        var nanSource = Matrix(1,1,MatrixLayout.RowMajor,[float.NaN]);
        var single = Matrix(1,1,MatrixLayout.RowMajor,[7]);
        single.Add(nanSource,0,0);
        TestAssert.True(float.IsNaN(single[0,0]), "Add's source scaling uses IEEE 0*NaN, unlike GEMM alpha=0");
        source[0,0] = float.NaN;
        t.Add(source,1);
        TestAssert.True(float.IsNaN(t[0,0]), "nonzero add scale propagates NaN");
    }

    private static void FanOut()
    {
        var gradient = new GradientBuffer(new FloatMatrix(2,2));
        gradient.Matrix.Fill(float.NaN);
        var first = Matrix(2,2,MatrixLayout.ColumnMajor,[1,2,3,4]);
        var second = Matrix(2,2,MatrixLayout.RowMajor,[10,20,30,40]);
        TestAssert.Equal(0f, gradient.Add(first), "first consumer overwrites");
        CheckMatrix(gradient.Matrix,[1,2,3,4]);
        TestAssert.Equal(1f, gradient.Add(second), "second consumer accumulates");
        CheckMatrix(gradient.Matrix,[11,22,33,44]);
        TestAssert.Equal(2, gradient.Contributions, "two contributions");
        gradient.BeginPass();
        TestAssert.Equal(0, gradient.Contributions, "reset contribution count");
        TestAssert.Equal(0f, gradient.Add(second), "new pass overwrites stale result");
        CheckMatrix(gradient.Matrix,[10,20,30,40]);
    }

    private static void FanOutProducts()
    {
        var g = new GradientBuffer(new FloatMatrix(2,1,MatrixLayout.ColumnMajor));
        g.Matrix.Fill(float.NaN);
        var a = Matrix(2,2,MatrixLayout.ColumnMajor,[1,2,3,4]);
        var b = Matrix(2,1,MatrixLayout.ColumnMajor,[2,1]);
        TestAssert.Equal(0f,g.AddProduct(a,b),"first product scale");
        CheckMatrix(g.Matrix,[4,10]);
        TestAssert.Equal(1f,g.AddProduct(a,b,0.5f),"second product scale");
        CheckMatrix(g.Matrix,[6,15]);
    }

    private static void RemoteOnlyMerge()
    {
        // E.g. hs2b (GPU1) has just fc1000 (GPU0) as consumer: no local contribution.
        // Original merge uses !doCopy=0 in this case (src/layer.cu:178,202-212,228-251).
        var remote = new GradientBuffer(new FloatMatrix(1,2));
        remote.Add(Matrix(1,2,MatrixLayout.RowMajor,[3,7]));
        var own = new GradientBuffer(new FloatMatrix(1,2));
        own.Matrix.Fill(float.NaN);
        TestAssert.Equal(0f,own.Add(remote.Matrix),"remote-only merge overwrites empty local buffer");
        CheckMatrix(own.Matrix,[3,7]);
    }

    private static void GraphFanOut(NetworkDefinition net)
    {
        var fanouts = net.Layers.Select(p => (p.Name, Consumers: net.Layers.Where(c => c.Inputs.Contains(p.Name)).ToArray()))
            .Where(p => p.Consumers.Length > 1).ToDictionary(p => p.Name, p => p.Consumers);
        string[] expected = ["data","pool2a","pool2b","pool3a_neuron","pool3b_neuron","hs1a","hs1b"];
        TestAssert.SequenceEqual(expected.Order().ToArray(), fanouts.Keys.Order().ToArray(), "all effective 145 fan-outs");
        var pairs = new Dictionary<string,string[]>
        {
            ["data"] = ["conv1a","conv1b"], ["pool2a"] = ["conv3a","conv3b"], ["pool2b"] = ["conv3a","conv3b"],
            ["pool3a_neuron"] = ["fc2048a","fc2048b"], ["pool3b_neuron"] = ["fc2048a","fc2048b"],
            ["hs1a"] = ["fc2048ba","fc2048bb"], ["hs1b"] = ["fc2048ba","fc2048bb"],
        };
        foreach (var p in fanouts)
        {
            TestAssert.SequenceEqual(pairs[p.Key],p.Value.Select(c=>c.Name).ToArray(),"fan-out consumers");
            TestAssert.SequenceEqual<int>([0,1],p.Value.Select(c=>c.Parameters["gpu"].AsInt()).ToArray(),"consumer devices");
            if (p.Key == "data") continue; // no trainable ancestor, no activation gradient requested
            // Original rcvdBInputs is per producer/device: both consumers use 0, then merge with 1.
            foreach (bool reverse in new[] { false, true })
            {
                var perDevice = new[] { new GradientBuffer(new FloatMatrix(1,2)), new GradientBuffer(new FloatMatrix(1,2)) };
                var consumers = reverse ? p.Value.Reverse() : p.Value.AsEnumerable();
                foreach (var consumer in consumers)
                {
                    int device = consumer.Parameters["gpu"].AsInt();
                    perDevice[device].Matrix.Fill(float.NaN);
                    var contribution = device == 0 ? Matrix(1,2,MatrixLayout.RowMajor,[1,2]) : Matrix(1,2,MatrixLayout.RowMajor,[10,20]);
                    TestAssert.Equal(0f,perDevice[device].Add(contribution),"each device's first consumer overwrites");
                }
                int ownDevice = net[p.Key].Parameters["gpu"].AsInt();
                TestAssert.Equal(1f,perDevice[ownDevice].Add(perDevice[1-ownDevice].Matrix),"merge accumulates remote contribution");
                CheckMatrix(perDevice[ownDevice].Matrix,[11,22]);
            }
        }
    }

    private static void Allocate145(NetworkDefinition net, JsonNode fixture)
    {
        int inner = fixture["inner_size"]!.GetValue<int>();
        var shapes = ShapeInference.Infer(net,DataDimensions.Jpeg(inner));
        var allocated = new List<LayerBuffers>(); // retain a complete network of real arrays together
        long weights = 0, biases = 0, activations = 0;
        int weightMatrices = 0, biasMatrices = 0, activationMatrices = 0;
        foreach (var expected in fixture["layers"]!.AsArray())
        {
            string name = expected!["name"]!.GetValue<string>();
            var layer = LayerBuffers.Allocate(shapes[name],3);
            allocated.Add(layer);
            int outputs = expected["outputs"]!.GetValue<int>();
            if (outputs == 0) TestAssert.True(layer.Activations is null,"cost has no public activation output");
            else
            {
                var m = layer.Activations!;
                bool trans = name.EndsWith("_neuron",StringComparison.Ordinal) || name.StartsWith("fc",StringComparison.Ordinal) || name == "probs";
                TestAssert.Equal((trans ? 3 : outputs,trans ? outputs : 3,trans),(m.Rows,m.Cols,m.IsTrans),name+" native layout");
                TestAssert.Equal(outputs*3,m.ElementCount,name+" activation elements");
                TestAssert.Equal(outputs*3,m.Storage.Length,name+" actual activation allocation");
                TestAssert.Equal(3,m.GetNumCases(trans),name+" cases");
                var featureCase = trans ? m.TransposeView() : m;
                featureCase[outputs-1,2] = 41;
                TestAssert.Equal(41f,m.Storage.Span[^1],name+" last case fastest");
                activations += m.ElementCount;
                activationMatrices++;
            }
            var expectedWeights = expected["weights_shapes"]!.AsArray();
            TestAssert.Equal(expectedWeights.Count,layer.Parameters.Weights.Count,name+" weight matrices");
            var inputs = expected["inputs"]!.AsArray();
            for (int i = 0; i < expectedWeights.Count; i++)
            {
                var w = layer.Parameters.Weights[i];
                int rows = expectedWeights[i]![0]!.GetValue<int>(), cols = expectedWeights[i]![1]!.GetValue<int>();
                TestAssert.Equal((rows,cols),(w.Matrix.Rows,w.Matrix.Cols),name+" W shape");
                TestAssert.Equal(rows*cols,w.Matrix.Storage.Length,name+" W actual storage");
                TestAssert.Equal(inputs[i]!.GetValue<string>(),w.InputName,name+" W input order");
                TestAssert.Equal(shapes[name].Type=="fc",w.Matrix.IsTrans,name+" W order");
                w.Matrix[rows-1,cols-1]=i+7;
                TestAssert.Equal((float)(i+7),w.Matrix.Storage.Span[^1],name+" W end writable");
                weights += w.Matrix.ElementCount;
                weightMatrices++;
            }
            var bd = expected["biases_shape"]!.AsArray();
            if (bd.Count == 0) TestAssert.True(layer.Parameters.Biases is null,name+" no bias buffer");
            else
            {
                var b = layer.Parameters.Biases!;
                int rows=bd[0]!.GetValue<int>(), cols=bd[1]!.GetValue<int>();
                TestAssert.Equal((rows,cols),(b.Rows,b.Cols),name+" bias shape");
                TestAssert.Equal(rows*cols,b.Storage.Length,name+" bias actual allocation");
                TestAssert.Equal(shapes[name].Type=="fc",b.IsTrans,name+" bias order");
                biases += b.ElementCount;
                biasMatrices++;
            }
        }
        TestAssert.Equal(49,allocated.Count,"all layers allocated");
        TestAssert.Equal(48,activationMatrices,"all non-cost activations");
        TestAssert.Equal(22,weightMatrices,"all W matrices");
        TestAssert.Equal(15,biasMatrices,"all biases");
        TestAssert.Equal(60954656L,weights,"hand weight total");
        TestAssert.Equal(10568L,biases,"hand bias total");
        TestAssert.Equal(60965224L,allocated.Sum(l=>l.Parameters.ParameterCount),"hand total allocated parameters");
        long expectedActivationElements = inner==224 ? 6026739 : 6038916;
        TestAssert.Equal(expectedActivationElements,activations,"hand activation total for 3 cases");
        var bb = allocated[shapes.Layers.ToList().FindIndex(l=>l.Name=="fc2048bb")].Parameters;
        TestAssert.SequenceEqual<string>(["hs1b","hs1a"],bb.Weights.Select(w=>w.InputName).ToArray(),"asymmetric FC input order retained");
        bb.Weights[0].Matrix[0,0]=123;
        TestAssert.Equal(0f,bb.Weights[1].Matrix[0,0],"two equal-shaped inputs have independent weight allocations");
    }

    private static void Validation()
    {
        Throws<ArgumentOutOfRangeException>(()=>new FloatMatrix(-1,2));
        Throws<OverflowException>(()=>new FloatMatrix(int.MaxValue,2));
        Throws<ArgumentOutOfRangeException>(()=>new FloatMatrix(1,2,(MatrixLayout)9));
        var m = new FloatMatrix(2,3);
        Throws<ArgumentOutOfRangeException>(()=>m[0,3]=1); // must not spill into next row
        Throws<ArgumentOutOfRangeException>(()=>m[2,0]=1);
        Throws<ArgumentOutOfRangeException>(()=>m.Slice(1,2,0,1));
        TestAssert.Equal(0,m.Slice(2,0,3,0).ElementCount,"empty end slice allowed");
        Throws<ArgumentException>(()=>m.Add(new FloatMatrix(3,2),0));
        Throws<ArgumentException>(()=>m.AddProduct(new FloatMatrix(2,2),new FloatMatrix(3,3),0));
        Throws<ArgumentException>(()=>m.AddProduct(m,new FloatMatrix(3,3),0));
        Throws<ArgumentException>(()=>m.Slice(0,2,0,2).Add(m.Slice(0,2,1,2),0));
        var l = new ActivationLayout(2,3,5,3);
        Throws<ArgumentOutOfRangeException>(()=>l.GetRow(0,0,5));
        Throws<ArgumentOutOfRangeException>(()=>l.GetFlatOffset(0,0,0,3));
        Throws<ArgumentOutOfRangeException>(()=>l.FromFlatOffset(l.ElementCount));
        Throws<ArgumentOutOfRangeException>(()=>l.FromMatrixIndex(l.Rows,0));
        Throws<ArgumentOutOfRangeException>(()=>new ActivationLayout(0,1,1,1));
        Throws<OverflowException>(()=>new ActivationLayout(3,100000,100000,1));
        Throws<ArgumentException>(()=>LayoutBridge.FromNchw(new float[2],1,1,1,3));
        var crop = new ActivationTensor(3,224,224,1);
        Throws<ArgumentException>(()=>LayoutBridge.ImportRgbCrop(new byte[1],crop,0));
        Throws<ArgumentOutOfRangeException>(()=>LayoutBridge.ImportRgbCrop(new byte[150528],crop,1));
        Throws<ArgumentException>(()=>LayoutBridge.ImportRgbCrop(new byte[150528],crop,0,new float[1]));
        Throws<ArgumentException>(()=>LayoutBridge.ImportRgbCrop(new byte[150528],new ActivationTensor(3,3,3,1),0));
        Throws<ArgumentOutOfRangeException>(()=>m.GetNumCases(false,0));
        Throws<ArgumentOutOfRangeException>(()=>m.GetNumCases(false,2,0));
        var g = new GradientBuffer(new FloatMatrix(1,1));
        Throws<ArgumentException>(()=>g.Add(new FloatMatrix(2,2)));
        TestAssert.Equal(0,g.Contributions,"failed gradient write leaves counter unchanged");
    }

    private static FloatMatrix Matrix(int rows,int cols,MatrixLayout layout,float[] logicalValues)
    {
        var m=new FloatMatrix(rows,cols,layout);
        for(int r=0;r<rows;r++)
            for(int c=0;c<cols;c++) m[r,c]=logicalValues[r*cols+c];
        return m;
    }

    private static void CheckMatrix(FloatMatrix m,float[] expected)
    {
        TestAssert.Equal(m.ElementCount,expected.Length,"hand matrix length");
        for(int r=0;r<m.Rows;r++)
            for(int c=0;c<m.Cols;c++) TestAssert.Equal(expected[r*m.Cols+c],m[r,c],$"hand matrix [{r},{c}]");
    }

    private static void Throws<T>(Action action) where T:Exception
    {
        try { action(); }
        catch(T) { return; }
        throw new TestFailureException("Expected "+typeof(T).Name);
    }
}
