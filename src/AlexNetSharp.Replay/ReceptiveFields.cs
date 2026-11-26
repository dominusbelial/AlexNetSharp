namespace AlexNetSharp.Replay;

public sealed record PixelRange(int X0,int Y0,int X1,int Y1)
{
    public bool Empty => X1 < X0 || Y1 < Y0;
    public override string ToString() => Empty ? "no contributing pixels" : $"x={X0}..{X1}, y={Y0}..{Y1} (inclusive)";
}
public sealed record ReceptiveField(PixelRange Contributing, PixelRange Nominal);

public static class ReceptiveFields
{
    public static ReceptiveField Compute(ReplayManifest manifest,string name,int x,int y)
    {
        var layers = manifest.Layers.ToDictionary(l => l.Name);
        var selected = layers[name];
        if (x < 0 || y < 0 || x >= selected.Shape[2] || y >= selected.Shape[1]) throw new ArgumentOutOfRangeException(nameof(x));
        PixelRange Walk(string layerName,PixelRange range,bool clip)
        {
            var l = layers[layerName];
            if (clip) range = new(Math.Max(0,range.X0),Math.Max(0,range.Y0),Math.Min(l.Shape[2]-1,range.X1),Math.Min(l.Shape[1]-1,range.Y1));
            if (range.Empty || l.Name == "data") return range;
            if (l.Inputs.Length == 0) throw new InvalidOperationException("Layer has no image ancestry.");
            if (l.Type is "fc" or "softmax" or "cost.logreg") throw new NotSupportedException("Select an image layer.");
            var previous = new PixelRange(range.X0*l.Stride+l.Start,range.Y0*l.Stride+l.Start,
                range.X1*l.Stride+l.Start+l.Kernel-1,range.Y1*l.Stride+l.Start+l.Kernel-1);
            var ranges = l.Inputs.Select(i => Walk(i,previous,clip)).Where(r => !r.Empty).ToArray();
            return ranges.Length == 0 ? new(0,0,-1,-1) : new(ranges.Min(r => r.X0),ranges.Min(r => r.Y0),ranges.Max(r => r.X1),ranges.Max(r => r.Y1));
        }
        var point = new PixelRange(x,y,x,y);
        return new(Walk(name,point,true),Walk(name,point,false));
    }
}
