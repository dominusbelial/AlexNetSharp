using System.Numerics;

using AlexNetSharp.Replay;

namespace AlexNetSharp.Viz;

public sealed record LayerBox(string Name, Vector3 Center, Vector3 Size, int Tower);
public sealed record LayerEdge(string Input, string Consumer, int Slot, bool CrossTower);
public sealed record ScreenPoint(string Name, float X, float Y);

/// <summary>Pure geometry, shared by GL rendering, HTML labels, tests and browser proof.</summary>
public sealed class LayerLayout
{
    public LayerBox[] Boxes { get; }
    public LayerEdge[] Edges { get; }
    public float Span { get; }
    public bool Portrait { get; }
    public LayerLayout(ReplayManifest manifest,bool portrait=false)
    {
        Portrait=portrait;
        var layers = manifest.Layers.ToDictionary(l => l.Name);
        var levels = new Dictionary<string,int>();
        int Level(string name)
        {
            if (levels.TryGetValue(name, out int level)) return level;
            var l = layers[name];
            return levels[name] = l.Inputs.Length == 0 ? 0 : l.Inputs.Max(Level) + 1;
        }
        int max = manifest.Layers.Max(l => Level(l.Name)); Span = portrait ? 32 : max * 2.15f + 5;
        Boxes = manifest.Layers.Select(l =>
        {
            float x = ((l.Name == "labels" ? max : Level(l.Name)) - max / 2f) * 2.15f;
            float y = l.Name == "labels" ? -5f : l.Name == "data" ? 0 : l.Tower == 1 ? -2.3f : 2.3f;
            if(portrait)
            {
                int stage=l.Name=="labels"?max:Level(l.Name),row=stage/3,col=stage%3;
                if(row%2==1)col=2-col; // snake the topological stages through the portrait canvas
                x=(col-1)*6.2f+(l.Name=="data"?0:l.Name=="labels"?-1.7f:l.Tower==1?-1.1f:1.1f);
                y=(max/3f/2-row)*3.3f+(l.Name=="labels"?-.9f:0);
            }
            // World X encodes W, Y encodes H, Z encodes C; logarithms keep the entire graph navigable.
            return new LayerBox(l.Name,new(x,y,0),new(.45f+.075f*MathF.Log2(Math.Max(1,l.Shape[2])),
                .45f+.12f*MathF.Log2(Math.Max(1,l.Shape[1])),.4f+.2f*MathF.Log2(Math.Max(1,l.Shape[0]))),l.Tower);
        }).ToArray();
        Edges = manifest.Layers.SelectMany(l => l.Inputs.Select((input,slot) =>
            new LayerEdge(input,l.Name,slot,layers[input].Tower >= 0 && l.Tower >= 0 && layers[input].Tower != l.Tower))).ToArray();
    }
    public ScreenPoint Project(LayerBox box, OrbitCamera camera, int width, int height)
    {
        Vector4 clip = Vector4.Transform(new Vector4(box.Center,1),camera.ViewProjection(width,height));
        return new(box.Name,(clip.X/clip.W+1)*width/2,(1-clip.Y/clip.W)*height/2);
    }
    public string? Pick(float x, float y, OrbitCamera camera, int width, int height)
    {
        if (!Matrix4x4.Invert(camera.ViewProjection(width,height),out var inverse)) return null;
        Vector3 Unproject(float z)
        {
            var p = Vector4.Transform(new Vector4(2*x/width-1,1-2*y/height,z,1),inverse);
            return new(p.X/p.W,p.Y/p.W,p.Z/p.W);
        }
        return PickRay(Unproject(0),Vector3.Normalize(Unproject(1)-Unproject(0)));
    }
    public string? PickRay(Vector3 origin, Vector3 direction)
    {
        float nearest = float.PositiveInfinity; string? name = null;
        foreach (var box in Boxes)
        {
            Vector3 min = box.Center-box.Size/2, max = box.Center+box.Size/2;
            float near = 0, far = float.PositiveInfinity;
            for (int axis = 0; axis < 3; axis++)
            {
                float d = direction[axis], o = origin[axis];
                if (Math.Abs(d) < 1e-7f) { if (o < min[axis] || o > max[axis]) { far = -1; break; } }
                else
                {
                    float a = (min[axis]-o)/d, b = (max[axis]-o)/d;
                    near = Math.Max(near,Math.Min(a,b)); far = Math.Min(far,Math.Max(a,b));
                }
            }
            if (far >= near && near < nearest) { nearest = near; name = box.Name; }
        }
        return name;
    }
}

public sealed class OrbitCamera
{
    private float span;
    private bool portrait;
    public OrbitCamera(float span){this.span=span;Reset();}
    public float Yaw { get; private set; }
    public float Pitch { get; private set; }
    public float Zoom { get; private set; }
    public Vector3 Target { get; private set; }
    public void Fit(float extent,bool isPortrait){span=extent;portrait=isPortrait;Reset();}
    public void Orbit(float dx,float dy) { Yaw = Math.Clamp(Yaw+dx,-1.05f,1.05f); Pitch = Math.Clamp(Pitch+dy,-1.05f,.8f); }
    public void Magnify(float amount) => Zoom = Math.Clamp(Zoom*MathF.Exp(amount),.6f,4);
    public void Pan(float dx,float dy) => Target = Vector3.Clamp(Target+new Vector3(dx,dy,0),new(-span/2),new(span/2));
    public void Reset() { Yaw = portrait ? -.16f : -.28f; Pitch = portrait ? -.24f : -.55f; Zoom = 1; Target = Vector3.Zero; }
    private float Distance(int width,int height)
    {
        float aspect=(float)width/height;
        return (portrait?Math.Max(span,18/aspect):Math.Max(20,span*1.32f/aspect)) / (2*MathF.Tan(.38f)) / Zoom;
    }
    // View depends on aspect through the fitted distance, so all callers use ViewFor with the same viewport.
    public Matrix4x4 ViewFor(int width,int height) => Matrix4x4.CreateLookAt(Target + Vector3.Transform(new Vector3(0,0,Distance(width,height)),
        Matrix4x4.CreateRotationX(Pitch)*Matrix4x4.CreateRotationY(Yaw)),Target,Vector3.UnitY);
    public Matrix4x4 Projection(int width,int height) => Matrix4x4.CreatePerspectiveFieldOfView(.76f,(float)width/height,.1f,500);
    public Matrix4x4 ViewProjection(int width,int height) => ViewFor(width,height)*Projection(width,height);
}
