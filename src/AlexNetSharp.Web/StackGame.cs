using AlexNetSharp.Replay;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using NMatrix=System.Numerics.Matrix4x4;

namespace AlexNetSharp.Web;

public sealed class StackGame : Game
{
    private readonly GraphicsDeviceManager graphics;
    private BasicEffect effect=null!;
    private SpriteBatch sprites=null!;
    private Texture2D white=null!,photo=null!,grid=null!,filters=null!;
    private readonly ReplaySnapshot snapshot;
    public LayerLayout Layout { get; }
    public OrbitCamera Camera { get; }
    public bool Started { get; set; }
    public string Selected { get; private set; }="conv1a";
    public string? Hovered { get; set; }
    public bool ShowFilters { get; set; }
    public ActivationPixel Pixel { get; set; }=new(0,0,0);
    public PixelImage GridImage { get; private set; }
    public int Width { get; private set; }=900;
    public int Height { get; private set; }=560;
    public int StackHeight=>Height/2;
    public ReplayLayer Layer=>snapshot.Manifest.Layers.Single(l=>l.Name==Selected);
    public ViewRect GridRectangle=>PixelViews.Fit(ShowFilters?filterImage:GridImage,PixelViews.GridRect(Width,Height));
    private readonly PixelImage filterImage;
    public StackGame(ReplaySnapshot replay)
    {
        snapshot=replay; Layout=new(replay.Manifest); Camera=new(Layout.Span);
        GridImage=PixelViews.Activations(snapshot,Layer); filterImage=PixelViews.Filters(snapshot);
        graphics=new(this); IsFixedTimeStep=false; IsMouseVisible=true;
    }
    private Texture2D Texture(PixelImage image)
    {
        var texture=new Texture2D(GraphicsDevice,image.Width,image.Height); texture.SetData(image.Rgba); return texture;
    }
    protected override void LoadContent()
    {
        effect=new BasicEffect(GraphicsDevice){VertexColorEnabled=true}; sprites=new(GraphicsDevice);
        white=new(GraphicsDevice,1,1); white.SetData(new[]{Color.White});
        photo=Texture(PixelViews.Input(snapshot)); grid=Texture(GridImage); filters=Texture(filterImage);
    }
    public void Select(string name)
    {
        var layer=snapshot.Manifest.Layers.Single(l=>l.Name==name);
        if(layer.Type is not ("conv" or "pool" or "cmrnorm" or "rnorm")) return;
        Selected=name; Pixel=new(0,0,0); ShowFilters=false; GridImage=PixelViews.Activations(snapshot,layer);
        if(Started){grid.Dispose();grid=Texture(GridImage);}
    }
    public void Size(int width,int height)
    {
        if(width==Width&&height==Height) return;
        Width=width;Height=height;graphics.PreferredBackBufferWidth=width;graphics.PreferredBackBufferHeight=height;
        // The DOM owns the canvas buffer; Draw synchronizes KNI presentation metadata.
    }
    private static Matrix Convert(NMatrix m)=>new(m.M11,m.M12,m.M13,m.M14,m.M21,m.M22,m.M23,m.M24,m.M31,m.M32,m.M33,m.M34,m.M41,m.M42,m.M43,m.M44);
    private static Vector3 Convert(System.Numerics.Vector3 v)=>new(v.X,v.Y,v.Z);
    protected override void Draw(GameTime time)
    {
        // Blazor window events can reset managed backbuffer dimensions to the browser window.
        // Keep them equal to this embedded canvas before KNI converts viewport Y to GL.
        GraphicsDevice.PresentationParameters.BackBufferWidth=Width;
        GraphicsDevice.PresentationParameters.BackBufferHeight=Height;
        GraphicsDevice.Clear(new Color(11,20,33));
        GraphicsDevice.Viewport=new(0,0,Width,Height);
        GraphicsDevice.DepthStencilState=DepthStencilState.Default; GraphicsDevice.BlendState=BlendState.Opaque;
        GraphicsDevice.RasterizerState=RasterizerState.CullNone;
        effect.World=Matrix.Identity; effect.View=Convert(Camera.View); effect.Projection=Convert(Camera.Projection(Width,StackHeight))*Matrix.CreateScale(1,.5f,1)*Matrix.CreateTranslation(0,.5f,0);
        var boxes=Layout.Boxes.ToDictionary(b=>b.Name);
        foreach(var edge in Layout.Edges)
        {
            Color color=edge.CrossTower?new Color(248,183,93):new Color(57,85,110);
            // Input slots are displaced vertically, making reversed fan-in observable.
            var a=Convert(boxes[edge.Input].Center);var b=Convert(boxes[edge.Consumer].Center);
            a.Z=-.4f;b.Z=-.4f;b.Y+=(edge.Slot==0?.18f:-.18f);
            VertexPositionColor[] line=[new(a,color),new(b,color)];
            foreach(var pass in effect.CurrentTechnique.Passes){pass.Apply();GraphicsDevice.DrawUserPrimitives(PrimitiveType.LineList,line,0,1);}
        }
        foreach(var box in Layout.Boxes)
        {
            var center=Convert(box.Center);var size=Convert(box.Size)/2;
            var layer=snapshot.Manifest.Layers.Single(l=>l.Name==box.Name);
            Color color=box.Name==Hovered?new Color(255,223,164):box.Name==Selected?new Color(105,245,202):layer.Type=="neuron"?new Color(189,148,236):box.Tower==1?new Color(93,148,213):new Color(64,191,170);
            Vector3[] p=[center+new Vector3(-size.X,-size.Y,-size.Z),center+new Vector3(size.X,-size.Y,-size.Z),center+new Vector3(size.X,size.Y,-size.Z),center+new Vector3(-size.X,size.Y,-size.Z),center+new Vector3(-size.X,-size.Y,size.Z),center+new Vector3(size.X,-size.Y,size.Z),center+new Vector3(size.X,size.Y,size.Z),center+new Vector3(-size.X,size.Y,size.Z)];
            int[] indices=[0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5];
            var vertices=indices.Select((i,k)=>new VertexPositionColor(p[i],k<12?color:new Color(color.R*.65f/255,color.G*.65f/255,color.B*.65f/255))).ToArray();
            foreach(var pass in effect.CurrentTechnique.Passes){pass.Apply();GraphicsDevice.DrawUserPrimitives(PrimitiveType.TriangleList,vertices,0,12);}
        }
        sprites.Begin(blendState:BlendState.AlphaBlend,samplerState:SamplerState.PointClamp);
        var r=GridRectangle; sprites.Draw(ShowFilters?filters:grid,new Rectangle(r.X,r.Y,r.Width,r.Height),Color.White);
        var pr=PixelViews.PhotoRect(Width,Height);sprites.Draw(photo,new Rectangle(pr.X,pr.Y,pr.Width,pr.Height),Color.White);
        if(!ShowFilters)
        {
            var range=ReceptiveFields.Compute(snapshot.Manifest,Selected,Pixel.X,Pixel.Y).Contributing;
            if(!range.Empty)
            {
                var rr=new Rectangle(pr.X+range.X0*pr.Width/224,pr.Y+range.Y0*pr.Height/224,
                    Math.Max(2,(range.X1-range.X0+1)*pr.Width/224),Math.Max(2,(range.Y1-range.Y0+1)*pr.Height/224));
                Border(rr,new Color(255,218,117));
            }
            int tile=Layer.Shape[1]+2,cols=PixelViews.Columns(Layer.Shape[0]);
            int px=(Pixel.Channel%cols*tile+Pixel.X+1)*r.Width/GridImage.Width+r.X,py=(Pixel.Channel/cols*tile+Pixel.Y+1)*r.Height/GridImage.Height+r.Y;
            Border(new(px-2,py-2,5,5),Color.White);
        }
        sprites.End();base.Draw(time);
    }
    private void Border(Rectangle r,Color color)
    {
        sprites.Draw(white,new Rectangle(r.X,r.Y,r.Width,2),color);sprites.Draw(white,new Rectangle(r.X,r.Bottom-2,r.Width,2),color);
        sprites.Draw(white,new Rectangle(r.X,r.Y,2,r.Height),color);sprites.Draw(white,new Rectangle(r.Right-2,r.Y,2,r.Height),color);
    }
    protected override void Dispose(bool disposing)
    {
        if(disposing){effect?.Dispose();sprites?.Dispose();white?.Dispose();photo?.Dispose();grid?.Dispose();filters?.Dispose();}
        base.Dispose(disposing);
    }
}
