using AlexNetSharp.Replay;

namespace AlexNetSharp.Viz;

public sealed record PixelImage(int Width,int Height,byte[] Rgba);
public sealed record ViewRect(int X,int Y,int Width,int Height);
public sealed record ActivationPixel(int Channel,int X,int Y);

public static class PixelViews
{
    public static PixelImage Input(ReplaySnapshot snapshot)
    {
        float[] data = snapshot.Values(snapshot.Manifest.Layers.Single(l => l.Name == "data").Activations);
        var rgba = new byte[224*224*4];
        for (int p = 0; p < 224*224; p++)
        {
            for (int c = 0; c < 3; c++) rgba[p*4+c] = (byte)Math.Clamp(data[c*224*224+p]+snapshot.Manifest.Image.Mean[c],0,255);
            rgba[p*4+3] = 255;
        }
        return new(224,224,rgba);
    }
    // Per-channel min/max; low midnight blue -> high mint. Constant maps use midpoint.
    public static PixelImage Activations(ReplaySnapshot snapshot,ReplayLayer layer)
    {
        int channels=layer.Shape[0], side=layer.Shape[1], cols=Columns(channels), rows=(channels+cols-1)/cols, tile=side+2;
        int width=cols*tile,height=rows*tile; var rgba=new byte[width*height*4];
        var data=snapshot.Values(layer.Activations);
        for(int c=0;c<channels;c++)
        {
            var plane=data.AsSpan(c*side*side,side*side); float min=float.PositiveInfinity,max=float.NegativeInfinity;
            foreach(float v in plane) { min=Math.Min(min,v); max=Math.Max(max,v); }
            for(int y=0;y<side;y++) for(int x=0;x<side;x++)
            {
                float t=max==min?.5f:(plane[y*side+x]-min)/(max-min);
                int p=((c/cols*tile+y+1)*width+c%cols*tile+x+1)*4;
                rgba[p]=(byte)(13+66*t); rgba[p+1]=(byte)(24+202*t); rgba[p+2]=(byte)(54+144*t); rgba[p+3]=255;
            }
        }
        return new(width,height,rgba);
    }
    public static PixelImage Filters(ReplaySnapshot snapshot)
    {
        var values=snapshot.Values(snapshot.Manifest.Conv1Filters); const int cols=16,tile=13,width=cols*tile,height=6*tile;
        var rgba=new byte[width*height*4];
        for(int f=0;f<96;f++)
        {
            var filter=values.AsSpan(f*363,363); float min=float.PositiveInfinity,max=float.NegativeInfinity;
            foreach(float v in filter) { min=Math.Min(min,v); max=Math.Max(max,v); }
            for(int y=0;y<11;y++) for(int x=0;x<11;x++)
            {
                int p=((f/cols*tile+y+1)*width+f%cols*tile+x+1)*4;
                for(int c=0;c<3;c++) rgba[p+c]=(byte)(255*(filter[c*121+y*11+x]-min)/(max-min));
                rgba[p+3]=255;
            }
        }
        return new(width,height,rgba);
    }
    public static int Columns(int channels)=>(int)Math.Ceiling(Math.Sqrt(channels*1.6));
    public static int StackHeight(int width,int height)=>width<=600?height-210:height/2;
    public static ViewRect GridRect(int width,int height)
    {
        int photo=Math.Min(224,Math.Max(120,width/4));
        return new(16,StackHeight(width,height)+42,Math.Max(80,width-photo-48),Math.Max(40,height-StackHeight(width,height)-58));
    }
    public static ViewRect PhotoRect(int width,int height)
    {
        int side=Math.Min(Math.Min(224,Math.Max(120,width/4)),height-StackHeight(width,height)-58);
        return new(width-side-16,StackHeight(width,height)+42,side,side);
    }
    public static ViewRect Fit(PixelImage image,ViewRect area)
    {
        float scale=Math.Min((float)area.Width/image.Width,(float)area.Height/image.Height);
        return new(area.X,area.Y,(int)(image.Width*scale),(int)(image.Height*scale));
    }
    public static ActivationPixel? Pick(float x,float y,ReplayLayer layer,ViewRect rect,PixelImage image)
    {
        if(x<rect.X||y<rect.Y||x>=rect.X+rect.Width||y>=rect.Y+rect.Height) return null;
        int ax=(int)((x-rect.X)*image.Width/rect.Width),ay=(int)((y-rect.Y)*image.Height/rect.Height);
        int side=layer.Shape[1],tile=side+2,channel=ay/tile*Columns(layer.Shape[0])+ax/tile,px=ax%tile-1,py=ay%tile-1;
        return channel<layer.Shape[0]&&px>=0&&py>=0&&px<side&&py<side?new(channel,px,py):null;
    }
}
