using System.Globalization;
using System.Text.Json.Nodes;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Core.Layers.Pool;
using AlexNetSharp.Core.Layers.Neurons;
using AlexNetSharp.Core.Shapes;
using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Tests;

internal static class PoolReluTests
{
    public static int RunAll(string repo, string original, bool full)
    {
        int count = Fixtures(repo) + Numerical() + Contracts() + Parity();
        count += RealGeometry(original, full);
        Console.WriteLine($"POOL/RELU TESTS PASSED: {count} fixture, numerical, contract, determinism and real-geometry groups.");
        return count;
    }
    private static int Fixtures(string repo)
    {
        var data = JsonNode.Parse(File.ReadAllText(Path.Combine(repo, "tests/fixtures/pool-relu-cases.json")))!;
        int count = 0;
        foreach (var node in data["pool"]!.AsArray())
        {
            var f = node!;
            string name = f["name"]!.GetValue<string>(), kind = f["pool"]!.GetValue<string>();
            int n = I(f, "cases");
            var g = new PoolGeometry(I(f,"size"), I(f,"channels"), 3, 2, I(f,"start"), I(f,"outputs_x"));
            var x = Matrix(g.InputRows,n,f["input"]!); var dy = Matrix(g.OutputRows,n,f["dy"]!);
            foreach (bool opt in new[] {false,true})
            {
                var p = new Engine(name,g,kind,opt);
                var y = new FloatMatrix(g.OutputRows,n); y.Fill(float.NaN);
                p.Forward(x,y); Check(f["forward"]!,y,name+" forward");
                var dx = new FloatMatrix(g.InputRows,n); dx.Fill(float.NaN);
                p.Backward(x,dy,y,dx); Check(f["dx"]!,dx,name+" overwrite NaN");
                dx.Fill(2); p.Backward(x,dy,y,dx,1); Check(f["dx"]!,dx,name+" add",2);
                dx.Fill(float.NaN); p.Backward(x,dy,y,dx,1);
                Each(dx,v=>TestAssert.True(float.IsNaN(v),name+" retain NaN"));
                if (kind == "avg")
                {
                    p.Backward(null,dy,null,dx); Check(f["dx"]!,dx,name+" avg no saved inputs");
                }
                count++;
            }
        }
        var relu = data["relu"]!;
        // Repeat into SIMD lanes plus tails, both matrix orders and in-place mappings.
        foreach (bool opt in new[] {false,true})
            foreach (var layout in new[] {MatrixLayout.RowMajor,MatrixLayout.ColumnMajor})
            {
                var x = new FloatMatrix(7,17,layout); var dy = new FloatMatrix(7,17,layout);
                for(int r=0;r<7;r++) for(int n=0;n<17;n++) { x[r,n]=F(relu["input"]![r]!); dy[r,n]=F(relu["dy"]![r]!); }
                var y = new FloatMatrix(7,17,layout);
                ReluForward(opt,x,y);
                for(int r=0;r<7;r++) for(int n=0;n<17;n++) Same(F(relu["forward"]![r]!),y[r,n],"relu forward special");
                var dx = new FloatMatrix(7,17,layout); dx.Fill(float.NaN);
                ReluBackward(opt,y,dy,dx);
                for(int r=0;r<7;r++) for(int n=0;n<17;n++) Same(F(relu["dx"]![r]!),dx[r,n],"relu backward special");
                dx.Fill(2); ReluBackward(opt,y,dy,dx,1);
                for(int r=0;r<7;r++) for(int n=0;n<17;n++) Same(F(relu["dx"]![r]!)+2,dx[r,n],"relu add");
                dx.Fill(float.NaN); ReluBackward(opt,y,dy,dx,1); Each(dx,v=>TestAssert.True(float.IsNaN(v),"relu retain NaN"));
                ReluForward(opt,x,x); Bits(y,x,"relu in-place forward");
                // Saved output is deliberately independent of any forward input.
                y.Fill(1); ReluBackward(opt,y,dy,dy); // upstream alias is safe
                for(int r=0;r<7;r++) for(int n=0;n<17;n++) Same(F(relu["dy"]![r]!),dy[r,n],"saved output mask and in-place gradient");
                y.Fill(0); dy.Fill(3); ReluBackward(opt,y,dy,dy); Each(dy,v=>Same(0,v,"zero output mask"));
                count++;
            }
        Console.WriteLine($"POOL/RELU FIXTURES: {data["pool"]!.AsArray().Count} independent pool cases, both engines; ReLU C/F, N=17, in-place and output mask.");
        return count;
    }

    private static int Numerical()
    {
        int checks = 0; double worst = 0;
        foreach(int seed in new[]{7701,7702,7703})
        {
            int size = seed == 7701 ? 5 : seed == 7702 ? 6 : 7;
            var g = new PoolGeometry(size,2,3,2,seed==7703 ? -1 : 0);
            var random = new Random(seed);
            var x = new FloatMatrix(g.InputRows,2); var dy = new FloatMatrix(g.OutputRows,2);
            // Unique exactly-representable spaced inputs; perturbations cannot switch a winner.
            var perm = Enumerable.Range(0,x.ElementCount).OrderBy(_=>random.Next()).ToArray();
            for(int i=0;i<perm.Length;i++) x[i/2,i%2]=perm[i]-perm.Length/2+0.25f;
            Fill(dy,random);
            foreach(bool opt in new[]{false,true})
            {
                var p = new Engine("numeric",g,"max",opt); var y = new FloatMatrix(g.OutputRows,2); var dx = new FloatMatrix(g.InputRows,2);
                p.Forward(x,y); p.Backward(x,dy,y,dx);
                double[] values = Enumerable.Range(0,x.ElementCount).Select(i=>(double)x[i/2,i%2]).ToArray();
                for(int i=0;i<values.Length;i++)
                {
                    double old=values[i]; const double eps=1e-5;
                    values[i]=old+eps; double plus=PoolLoss(values,dy,g);
                    values[i]=old-eps; double minus=PoolLoss(values,dy,g); values[i]=old;
                    Verify((plus-minus)/(2*eps),dx[i/2,i%2]);
                }
                ReluForward(opt,x,x); ReluBackward(opt,x,newOnes(x),dx);
                // A separate nonzero signed array checks ReLU central differences.
                for(int i=0;i<values.Length;i++)
                {
                    double v=values[i]; const double eps=1e-5;
                    double numeric=(Math.Max(0,v+eps)-Math.Max(0,v-eps))/(2*eps);
                    Verify(numeric,dx[i/2,i%2]);
                }
                // Restore original nonzero input for the other engine.
                for(int i=0;i<values.Length;i++) x[i/2,i%2]=(float)values[i];
            }
        }
        Console.WriteLine($"POOL/RELU NUMERICAL: {checks} float64 central differences, epsilon=1e-5, tolerance=2e-6*(1+abs(numeric)), worst={worst:G8}.");
        return 6;
        void Verify(double expected,float actual)
        {
            double error=Math.Abs(expected-actual); worst=Math.Max(worst,error); checks++;
            TestAssert.True(double.IsFinite(expected)&&float.IsFinite(actual)&&error<=2e-6*(1+Math.Abs(expected)),"finite difference");
        }
        static FloatMatrix newOnes(FloatMatrix x) { var m=new FloatMatrix(x.Rows,x.Cols);m.Fill(1);return m; }
    }
    private static double PoolLoss(double[] x, FloatMatrix dy, PoolGeometry g)
    {
        double loss=0;
        for(int row=0;row<dy.Rows;row++) for(int n=0;n<dy.Cols;n++)
        {
            int c=row/(g.OutputSize*g.OutputSize),oy=row/g.OutputSize%g.OutputSize,ox=row%g.OutputSize;
            double max=-2e38;
            // Enumerate the whole input and test set membership, independently of undo bounds.
            for(int y=0;y<g.ImageSize;y++) for(int xx=0;xx<g.ImageSize;xx++)
                if(y>=g.Start+oy*g.Stride && y<g.Start+oy*g.Stride+g.SizeX && xx>=g.Start+ox*g.Stride && xx<g.Start+ox*g.Stride+g.SizeX)
                    max=Math.Max(max,x[((c*g.ImageSize+y)*g.ImageSize+xx)*dy.Cols+n]);
            loss+=max*dy[row,n];
        }
        return loss;
    }

    private static int Parity()
    {
        for(int seed=77100;seed<77136;seed++)
        {
            var rng=new Random(seed); int size=rng.Next(3,10), k=rng.Next(1,4), stride=rng.Next(1,k+1);
            var g=new PoolGeometry(size,rng.Next(1,4),k,stride,seed%3-1);
            int cases=new[]{1,3,8,17,33,128}[seed%6];
            foreach(string kind in new[]{"max","avg"})
            {
                var x=new FloatMatrix(g.InputRows,cases); var dy=new FloatMatrix(g.OutputRows,cases); Fill(x,rng);Fill(dy,rng);
                // Deliberate ties, NaNs, signed zeros, infinities in vector lanes and tails.
                float[] specials=[-0f,0f,float.NaN,float.NegativeInfinity,-3e38f,float.PositiveInfinity];
                for(int i=0;i<x.ElementCount;i+=11) x[i/cases,i%cases]=specials[(i/11)%specials.Length];
                for(int i=0;i<dy.ElementCount;i+=53) dy[i/cases,i%cases]=float.NaN;
                var reference=new ScalarPool("parity",g,kind); var y=new FloatMatrix(g.OutputRows,cases);reference.Forward(x,y);
                var upstream=new FloatMatrix(x.Rows,cases);Fill(upstream,rng); upstream[0,0]=float.NaN;
                foreach(float scale in new[]{0f,1f,0.5f})
                {
                    var dx=new FloatMatrix(g.InputRows,cases);dx.Fill(scale == 0 || seed % 2 == 0 ? float.NaN : 2.5f);reference.Backward(x,dy,y,dx,scale);
                    foreach(int threads in new[]{1,2,Environment.ProcessorCount,Environment.ProcessorCount})
                    {
                        var fast=new OptimizedPool("parity",g,kind,threads); var fy=new FloatMatrix(y.Rows,cases);var fdx=new FloatMatrix(dx.Rows,cases);fdx.Fill(scale == 0 || seed % 2 == 0 ? float.NaN : 2.5f);
                        fast.Forward(x,fy);Bits(y,fy,"pool forward parity");fast.Backward(x,dy,fy,fdx,scale);Bits(dx,fdx,"pool backward parity");
                        var ry=new FloatMatrix(x.Rows,cases);var fry=new FloatMatrix(x.Rows,cases);ScalarRelu.Forward(x,ry);OptimizedRelu.Forward(x,fry,threads);Bits(ry,fry,"relu forward parity");
                        var rd=new FloatMatrix(x.Rows,cases);var frd=new FloatMatrix(x.Rows,cases);rd.Fill(scale == 0 || seed % 2 == 0 ? float.NaN : 2.5f);frd.Fill(scale == 0 || seed % 2 == 0 ? float.NaN : 2.5f);
                        ScalarRelu.Backward(ry,upstream,rd,scale);OptimizedRelu.Backward(fry,upstream,frd,scale,threads);Bits(rd,frd,"relu backward parity");
                    }
                }
            }
        }
        Console.WriteLine("POOL/RELU PARITY: 36 seeded geometries, max/avg, N=1..128, scales 0/1/0.5, threads 1/2/all/repeated-all, raw bits.");
        return 36;
    }

    private static int Contracts()
    {
        foreach(bool opt in new[]{false,true})
        {
            var g=new PoolGeometry(5,1,3,2); var p=new Engine("named-pool",g,"max",opt);
            foreach(string kind in new[]{"rand","maxabs","bogus"})
                Throws(()=>new Engine("named-pool",g,kind,opt),"named-pool","unsupported in this port");
            var backing=new FloatMatrix(29,20);backing.Fill(1234);
            var x=backing.Slice(2,25,1,17);Fill(x,new Random(7));
            var ybase=new FloatMatrix(8,20);ybase.Fill(1234);var y=ybase.Slice(2,4,1,17);
            var dy=new FloatMatrix(4,17);dy.Fill(1);var dxbase=new FloatMatrix(29,20);dxbase.Fill(1234);var dx=dxbase.Slice(2,25,1,17);
            p.Forward(x,y);p.Backward(x,dy,y,dx);
            var cx=new FloatMatrix(25,17);for(int r=0;r<25;r++)for(int n=0;n<17;n++)cx[r,n]=x[r,n];
            var cy=new FloatMatrix(4,17);var cd=new FloatMatrix(25,17);new ScalarPool("copy",g).Forward(cx,cy);new ScalarPool("copy",g).Backward(cx,dy,cy,cd);
            Bits(cy,y,"strided output");Bits(cd,dx,"strided dx");
            Sentinels(ybase,2,4,1,17);Sentinels(dxbase,2,25,1,17);Sentinels(backing,2,25,1,17);
            Throws(()=>p.Forward(x,new FloatMatrix(4,17,MatrixLayout.ColumnMajor)),"canonical");
            Throws(()=>p.Forward(x,backing.Slice(0,4,0,17)),"storage");
            Throws(()=>p.Backward(x,dy,y,x),"storage");
            Throws(()=>p.Backward(x,new FloatMatrix(3,17),y,dx),"canonical");
            var rbase=new FloatMatrix(10,24);rbase.Fill(1234);var rx=rbase.Slice(1,7,2,17);Fill(rx,new Random(9));
            var ry=new FloatMatrix(7,17);ReluForward(opt,rx,ry);ReluForward(opt,rx,rx);Bits(ry,rx,"strided in-place relu");Sentinels(rbase,1,7,2,17);
            Throws(()=>ReluForward(opt,rbase.Slice(0,7,0,17),rx),"overlapping");
            var plane=new ActivationTensor(1,3,3,2);plane.Matrix.Fill(-2);plane[0,1,1,1]=3;
            var planeView=plane.ChannelPlaneView(0,1);var planeResult=new FloatMatrix(3,3,MatrixLayout.ColumnMajor);
            ReluForward(opt,planeView,planeResult);
            for(int r=0;r<3;r++)for(int n=0;n<3;n++)Same(r==1&&n==1?3:0,planeResult[r,n],"mixed layout noncontiguous plane");
            var planeGrad=new FloatMatrix(3,3);planeGrad.Fill(2);
            ReluBackward(opt,planeResult,planeGrad,planeView);
            for(int r=0;r<3;r++)for(int n=0;n<3;n++)Same(r==1&&n==1?2:0,planeView[r,n],"strided gradient result");
            for(int r=0;r<9;r++)Same(-2,plane.Matrix[r,0],"untouched case in strided ReLU");
            var output=new FloatMatrix(1,1);output.Fill(1);var grad=new FloatMatrix(1,1);grad.Fill(3);var target=new FloatMatrix(1,1);target.Fill(2);
            ReluBackward(opt,output,grad,target,0.5f);Same(5,target[0,0],"relu positive scale is boolean add");
            target.Fill(float.NaN);ReluBackward(opt,output,grad,target,-1);Same(3,target[0,0],"relu nonpositive overwrites");
            // Changed saved output proves the mask source independently of any original input.
            output.Fill(-1);ReluBackward(opt,output,grad,target);Same(0,target[0,0],"mask negative saved output");
        }
        Throws(()=>new ScalarPool("bad",new(5,1,3,0)),"geometry");
        Throws(()=>new OptimizedPool("bad",new(5,1,3,2),maxDegreeOfParallelism:0),"maxDegree");
        return 3;
    }

    private static int RealGeometry(string original,bool full)
    {
        var network=ShapeTests.Parse145(original);var shapes=ShapeInference.Infer(network);int count=0;
        foreach(var layer in network.Layers.Where(l=>l.Type=="pool"))
        {
            var s=shapes[layer.Name];var scalar=ScalarPool.FromLayer(layer,s);var g=scalar.Geometry;var fast=OptimizedPool.FromLayer(layer,s);
            int size=g.ImageSize, expected=size==55?27:size==27?13:6;
            TestAssert.Equal(expected,g.OutputSize,layer.Name+" shape");
            int start=(expected-1)*2;TestAssert.Equal(size-3,start,layer.Name+" last window");
            var x=new FloatMatrix(g.InputRows,3);var y=new FloatMatrix(g.OutputRows,3);var dy=new FloatMatrix(g.OutputRows,3);var dx=new FloatMatrix(g.InputRows,3);
            Fill(x,new Random(14570+count));Fill(dy,new Random(14580+count));fast.Forward(x,y);fast.Backward(x,dy,y,dx);
            var sy=new FloatMatrix(y.Rows,3);var sd=new FloatMatrix(dx.Rows,3);scalar.Forward(x,sy);scalar.Backward(x,dy,sy,sd);Bits(sy,y,"real forward");Bits(sd,dx,"real backward");
            if(full)
                foreach(int threads in new[]{1,2,Environment.ProcessorCount,Environment.ProcessorCount})
                {
                    var p=OptimizedPool.FromLayer(layer,s,threads);p.Forward(x,y);p.Backward(x,dy,y,dx);Bits(sy,y,"full real forward threads");Bits(sd,dx,"full real backward threads");
                }
            // Each tap of the final window is the unique max in turn. Only that
            // tap gets the final output's gradient. Right and bottom bounds both proven.
            for(int yy=start;yy<size;yy++)for(int xx=start;xx<size;xx++)
            {
                x.Fill(-10);x[yy*size+xx,0]=10;dy.Fill(0);dy[expected*expected-1,0]=7;
                fast.Forward(x,y);Same(10,y[expected*expected-1,0],"last-window tap forward");fast.Backward(x,dy,y,dx);
                for(int r=0;r<size*size;r++)Same(r==yy*size+xx?7:0,dx[r,0],"last-window exact membership");
            }
            // All outside pixels are larger: any accidental read beyond the
            // last window would change its max. Equal inside pixels then prove
            // the entire backward membership set, including exclusion of outsiders.
            x.Fill(10);dy.Fill(0);dy[expected*expected-1,0]=7;
            for(int yy=start;yy<size;yy++)for(int xx=start;xx<size;xx++)x[yy*size+xx,0]=1;
            fast.Forward(x,y);Same(1,y[expected*expected-1,0],"last-window excludes all outside pixels");
            fast.Backward(x,dy,y,dx);
            for(int r=0;r<size*size;r++)Same(r/size>=start&&r%size>=start?7:0,dx[r,0],"last-window complete tie membership");
            if(layer.Name.StartsWith("pool2",StringComparison.Ordinal))
                TestAssert.True(network.Layers.All(l=>l.Name!=layer.Name+"_neuron"),"pool2 no implicit ReLU");
            else
            {
                var neuron=network.Layers.Single(l=>l.Name==layer.Name+"_neuron");
                TestAssert.Equal("relu",((NeuronLayerParams)neuron.TypedParams).NeuronType,"implicit relu type");
                TestAssert.Equal(layer.Name,neuron.Inputs.Single(),"implicit relu follows pool");
                x.Fill(-3);fast.Forward(x,y);Each(y,v=>Same(-3,v,"pool remains signed"));
                var relu=new FloatMatrix(y.Rows,y.Cols);OptimizedRelu.Forward(y.TransposeView(),relu.TransposeView());Each(relu,v=>Same(0,v,"post-pool ReLU"));
                OptimizedRelu.Backward(relu.TransposeView(),dy.TransposeView(),dy.TransposeView());fast.Backward(x,dy,y,dx);Each(dx,v=>Same(0,v,"post-pool ReLU backward"));
            }
            count++;
        }
        // Every configured neuron is detached and all consumers are rewired.
        string[] expectedNames=["pool1a","pool1b","conv2a","conv2b","conv3a","conv3b","conv4a","conv4b","pool3a","pool3b","fc2048a","fc2048b","fc2048ba","fc2048bb"];
        foreach(string name in expectedNames)
        {
            var neuron=network.Layers.Single(l=>l.Name==name+"_neuron");
            TestAssert.Equal(name,neuron.Inputs.Single(),"all detached neurons");
            TestAssert.Equal("relu",((NeuronLayerParams)neuron.TypedParams).NeuronType,"all neurons relu");
            TestAssert.True(network.Layers.Where(l=>l.Name!=neuron.Name).All(l=>!l.Inputs.Contains(name)),"consumers use detached output");
        }
        Console.WriteLine("POOL REAL GEOMETRY: six 145 layers at N=3; 55->27 (52..54), 27->13 (24..26), 13->6 (10..12); all nine final taps and 14 detached ReLUs.");
        return count+1;
    }

    internal sealed class Engine
    {
        private readonly ScalarPool? _scalar;private readonly OptimizedPool? _fast;
        internal Engine(string name,PoolGeometry g,string kind,bool optimized,int threads=-1)
        {if(optimized)_fast=new(name,g,kind,threads);else _scalar=new(name,g,kind);}
        internal void Forward(FloatMatrix x,FloatMatrix y){if(_fast!=null)_fast.Forward(x,y);else _scalar!.Forward(x,y);}
        internal void Backward(FloatMatrix? x,FloatMatrix dy,FloatMatrix? y,FloatMatrix dx,float scale=0){if(_fast!=null)_fast.Backward(x,dy,y,dx,scale);else _scalar!.Backward(x,dy,y,dx,scale);}
    }
    internal static void ReluForward(bool opt,FloatMatrix x,FloatMatrix y){if(opt)OptimizedRelu.Forward(x,y);else ScalarRelu.Forward(x,y);}
    internal static void ReluBackward(bool opt,FloatMatrix y,FloatMatrix dy,FloatMatrix dx,float scale=0){if(opt)OptimizedRelu.Backward(y,dy,dx,scale);else ScalarRelu.Backward(y,dy,dx,scale);}
    internal static void Fill(FloatMatrix m,Random rng){for(int r=0;r<m.Rows;r++)for(int n=0;n<m.Cols;n++)m[r,n]=(rng.Next(-16,17))/4f;}
    private static int I(JsonNode n,string key)=>n[key]!.GetValue<int>();
    private static float F(JsonNode n)=>n.ToJsonString().StartsWith('"') ? n.GetValue<string>() switch {"NaN"=>float.NaN,"-0"=>-0f,"Infinity"=>float.PositiveInfinity,"-Infinity"=>float.NegativeInfinity,_=>throw new Exception("bad fixture value")} : float.Parse(n.ToJsonString(),CultureInfo.InvariantCulture);
    private static FloatMatrix Matrix(int rows,int cols,JsonNode data){var m=new FloatMatrix(rows,cols);for(int i=0;i<rows*cols;i++)m[i/cols,i%cols]=F(data[i]!);return m;}
    private static void Check(JsonNode data,FloatMatrix m,string label,float add=0){for(int i=0;i<m.ElementCount;i++){float expected=F(data[i]!);Same(add==0?expected:expected+add,m[i/m.Cols,i%m.Cols],label);}}
    private static void Same(float expected,float actual,string label)
    {
        if(float.IsNaN(expected))TestAssert.True(float.IsNaN(actual),label+" expected NaN");
        else TestAssert.Equal(BitConverter.SingleToInt32Bits(expected),BitConverter.SingleToInt32Bits(actual),label+" bits");
    }
    internal static void Bits(FloatMatrix expected,FloatMatrix actual,string label)
    {for(int r=0;r<expected.Rows;r++)for(int n=0;n<expected.Cols;n++)TestAssert.Equal(BitConverter.SingleToInt32Bits(expected[r,n]),BitConverter.SingleToInt32Bits(actual[r,n]),label);}
    private static void Each(FloatMatrix m,Action<float> check){for(int r=0;r<m.Rows;r++)for(int n=0;n<m.Cols;n++)check(m[r,n]);}
    private static void Throws(Action f,params string[] messages)
    {try{f();}catch(ArgumentException e){foreach(string msg in messages)TestAssert.True(e.Message.Contains(msg),"error "+msg);return;}catch(NotSupportedException e){foreach(string msg in messages)TestAssert.True(e.Message.Contains(msg),"error "+msg);return;}throw new TestFailureException("expected rejection");}
    private static void Sentinels(FloatMatrix m,int r0,int rows,int n0,int cols)
    {for(int r=0;r<m.Rows;r++)for(int n=0;n<m.Cols;n++)if(r<r0||r>=r0+rows||n<n0||n>=n0+cols)Same(1234,m[r,n],"sentinel");}
}
