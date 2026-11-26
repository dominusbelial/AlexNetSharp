using System.Text.Json;
using AlexNetSharp.Replay;
using AlexNetSharp.Viz;
using FontStashSharp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

namespace AlexNetSharp.Desktop;

public sealed class DesktopGame : StackGame
{
    private readonly ReplaySnapshot replay;
    private readonly string? proof;
    private readonly Dictionary<string,ExplainCard> cards;
    private readonly List<string> pinned=[];
    private FontSystem fonts=null!;
    private SpriteBatch ui=null!;
    private Texture2D white=null!;
    private MouseState previousMouse;
    private KeyboardState previousKeyboard;
    private int frame,drag;
    private double? firstFrameMs;
    private readonly List<object> receipts=[];
    private bool deeper;
    private float cardScroll;
    private int cardContentHeight;
    private readonly List<(string Id,Rectangle Close,Rectangle[] Links)> cardHits=[];
    public DesktopGame(ReplaySnapshot replay,string? proof):base(replay)
    {
        this.replay=replay;this.proof=proof;EmbeddedCanvas=false;
        graphics.PreferredBackBufferWidth=1100;graphics.PreferredBackBufferHeight=720;
        Window.Title="AlexNet / 2012 - replay explorer";
        Window.AllowUserResizing=true;
        cards=Directory.GetFiles(Path.Combine(AppContext.BaseDirectory,"content/explain"),"*.md").Select(p=>ExplainCard.Parse(File.ReadAllText(p))).ToDictionary(c=>c.Id);
        if(proof is not null)Directory.CreateDirectory(proof);
    }
    protected override void LoadContent()
    {
        base.LoadContent();Started=true;
        fonts=new();fonts.AddFont(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory,"content/fonts/DejaVuSans.ttf")));
        ui=new(GraphicsDevice);white=new(GraphicsDevice,1,1);white.SetData(new[]{Color.White});
    }
    private void Pin(string name){if(!pinned.Contains(name))pinned.Add(name);Select(name);}
    protected override void Update(GameTime gameTime)
    {
        int w=GraphicsDevice.PresentationParameters.BackBufferWidth,h=GraphicsDevice.PresentationParameters.BackBufferHeight;
        Size(Math.Max(300,w-340),h-48);
        var mouse=Mouse.GetState();var keyboard=Keyboard.GetState();
        bool down=mouse.LeftButton==ButtonState.Pressed||mouse.RightButton==ButtonState.Pressed;
        bool wasDown=previousMouse.LeftButton==ButtonState.Pressed||previousMouse.RightButton==ButtonState.Pressed;
        if(down&&!wasDown)drag=0;
        if(down&&wasDown&&mouse.X<Width)
        {
            int dx=mouse.X-previousMouse.X,dy=mouse.Y-previousMouse.Y;drag+=Math.Abs(dx)+Math.Abs(dy);
            if(mouse.RightButton==ButtonState.Pressed||keyboard.IsKeyDown(Keys.LeftShift))Camera.Pan(-dx*.06f,dy*.06f);else Camera.Orbit(dx*.006f,dy*.006f);
        }
        if(mouse.X<Width&&mouse.Y<StackHeight)Hovered=Layout.Pick(mouse.X,mouse.Y,Camera,Width,StackHeight);else Hovered=null;
        if(!down&&wasDown&&drag<4)
        {
            if(Hovered is not null)Pin(Hovered);
            if(!ShowFilters){var pixel=PixelViews.Pick(mouse.X,mouse.Y,Layer,GridRectangle,GridImage);if(pixel is not null)Pixel=pixel;}
            if(mouse.X>=Width+18 && mouse.Y>=70 && mouse.Y<108)ShowFilters=!ShowFilters;
            if(mouse.X>=Width+18 && mouse.Y>=112 && mouse.Y<145)Camera.Reset();
            if(mouse.X>=Width+18 && mouse.Y>=150 && mouse.Y<184)deeper=!deeper;
            if(mouse.Y>=205)
                foreach(var hit in cardHits.ToArray())
                {
                    if(hit.Close.Contains(mouse.X,mouse.Y)){pinned.Remove(hit.Id);break;}
                    for(int link=0;link<hit.Links.Length;link++)if(hit.Links[link].Contains(mouse.X,mouse.Y)){Pin(cards[hit.Id].Links[link]);break;}
                }
        }
        if(mouse.ScrollWheelValue!=previousMouse.ScrollWheelValue)
        {
            int delta=mouse.ScrollWheelValue-previousMouse.ScrollWheelValue;
            if(mouse.X>=Width)cardScroll=Math.Clamp(cardScroll-delta*.5f,0,Math.Max(0,cardContentHeight-h+220));
            else Camera.Magnify(delta*.001f);
        }
        foreach(var key in keyboard.GetPressedKeys())
        {
            if(key is Keys.Left or Keys.Right or Keys.Up or Keys.Down or Keys.A or Keys.D or Keys.W or Keys.S)
                CameraControls.Key(Camera,key.ToString());
            else if(!previousKeyboard.IsKeyDown(key))
            {
                if(key==Keys.Escape)Exit();
                else if(key==Keys.F)ShowFilters=!ShowFilters;
                else if(key==Keys.Tab)deeper=!deeper;
                else CameraControls.Key(Camera,key is Keys.OemPlus or Keys.Add?"+":key is Keys.OemMinus or Keys.Subtract?"-":key.ToString());
            }
        }
        previousMouse=mouse;previousKeyboard=keyboard;
        if(proof is not null)
        {
            if(frame>=12&&frame<24){var p=Layout.Project(Layout.Boxes.Single(b=>b.Name=="conv1a"),Camera,Width,StackHeight);Hovered=Layout.Pick(p.X,p.Y,Camera,Width,StackHeight);if(Hovered!="conv1a")throw new InvalidOperationException("Proof hover picking failed");}
            if(frame==24)Pin("conv1a");
            if(frame==36){var r=GridRectangle;var p=PixelViews.Pick(r.X+4.5f*r.Width/GridImage.Width,r.Y+5.5f*r.Height/GridImage.Height,Layer,r,GridImage);Pixel=p??throw new InvalidOperationException("Proof pixel pick failed");}
            if(frame==48)Camera.Orbit(.3f,.13f);
            if(frame==60)ShowFilters=true;
        }
        base.Update(gameTime);
    }
    private void Text(string text,int x,int y,int size,Color color)=>ui.DrawString(fonts.GetFont(size),text,new Vector2(x,y),color);
    private string Wrap(string text,int size,int width)
    {
        var font=fonts.GetFont(size);var lines=new List<string>();
        foreach(string paragraph in text.Replace("\r","").Split('\n'))
        {
            string line="";
            foreach(string word in paragraph.Split(' ',StringSplitOptions.RemoveEmptyEntries))
            {
                if(line.Length>0&&font.MeasureString(line+" "+word).X>width){lines.Add(line);line=word;}else line+=(line.Length==0?"":" ")+word;
            }
            lines.Add(line);
        }
        return string.Join('\n',lines);
    }
    protected override void Draw(GameTime gameTime)
    {
        base.Draw(gameTime);
        int w=GraphicsDevice.PresentationParameters.BackBufferWidth,h=GraphicsDevice.PresentationParameters.BackBufferHeight;
        GraphicsDevice.Viewport=new(0,0,w,h);
        ui.Begin(samplerState:SamplerState.LinearClamp);
        Color muted=new(151,178,195),mint=new(138,224,190);
        Text("ALEXNET / 2012",18,12,19,mint);
        Text("Untrained weights - top-5 predictions are meaningless",18,38,13,Color.Wheat);
        Text(Hovered is not null?Hovered+" - "+cards[Hovered].Summary:"Drag: orbit / Shift: pan / wheel: zoom / arrows, WASD, +/- / R: reset",18,StackHeight-28,13,Color.Wheat);
        Text(ShowFilters?"96 CONV1 RGB FILTERS":"ACTIVATIONS / "+Selected,18,StackHeight+12,14,muted);
        var pr=PixelViews.PhotoRect(Width,Height);Text("INPUT CROP",pr.X,StackHeight+12,13,muted);
        ui.Draw(white,new Rectangle(Width,0,w-Width,h),new Color(22,37,53));
        int x=Width+18;
        Text("EXPLAIN / COMPARE",x,20,18,mint);
        Text("[F] Activations / filters",x,78,15,Color.White);
        Text("[R] Reset view",x,118,15,Color.White);
        Text("[Tab] Deeper / formula",x,158,15,Color.White);
        Text("Scroll here to read cards / click links to compare",x,185,11,muted);
        if(pinned.Count==0)Text("Click a layer to pin its card.",x,214,15,muted);
        ui.End();
        GraphicsDevice.ScissorRectangle=new Rectangle(Width,205,w-Width,h-215);
        using var clipped=new RasterizerState{ScissorTestEnable=true};
        ui.Begin(samplerState:SamplerState.LinearClamp,rasterizerState:clipped);
        cardHits.Clear();
        int cardY=210-(int)cardScroll;
        foreach(string id in pinned)
        {
            var card=cards[id];var layer=replay.Manifest.Layers.FirstOrDefault(l=>l.Name==card.Id);
            string summary=Wrap(card.Summary,14,286);
            string values=layer is null?"Layer type":$"C x H x W: {string.Join(" x ",layer.Shape)}\nParameters: {layer.ParameterCount:N0}\nmin {layer.Statistics.Min:G4} / max {layer.Statistics.Max:G4}\nmean {layer.Statistics.Mean:G4} / std {layer.Statistics.StandardDeviation:G4}";
            string body=Wrap(CardText.PlainBody(card.Body)+(deeper?"\n\nFormula: "+card.Formula:""),12,286);
            string citations=Wrap("Paper "+card.PaperSection+" / "+string.Join("; ",card.Citations),10,286);
            int summaryY=cardY+34,valuesY=summaryY+(int)fonts.GetFont(14).MeasureString(summary).Y+14;
            int bodyY=valuesY+(int)fonts.GetFont(13).MeasureString(values).Y+14;
            int linksY=bodyY+(int)fonts.GetFont(12).MeasureString(body).Y+14;
            int citationY=linksY+card.Links.Length*24+8;
            int cardHeight=citationY-cardY+(int)fonts.GetFont(10).MeasureString(citations).Y+18;
            ui.Draw(white,new Rectangle(x-4,cardY-5,308,cardHeight),new Color(32,58,65));
            Text(card.Title,x,cardY,22,mint);Text("x",w-40,cardY,18,Color.White);
            Text(summary,x,summaryY,14,Color.White);Text(values,x,valuesY,13,muted);
            Text(body,x,bodyY,12,Color.White);Text(citations,x,citationY,10,muted);
            var links=card.Links.Select((link,i)=>new Rectangle(x,linksY+i*24,285,22)).ToArray();
            for(int i=0;i<links.Length;i++)Text(card.Links[i]+" >",x,links[i].Y,12,mint);
            cardHits.Add((id,new Rectangle(w-50,cardY,35,30),links));
            cardY+=cardHeight+16;
        }
        cardContentHeight=cardY+(int)cardScroll-210;
        ui.End();
        ui.Begin(samplerState:SamplerState.LinearClamp);
        var rf=ReceptiveFields.Compute(replay.Manifest,Selected,Pixel.X,Pixel.Y);
        Text($"Pixel c={Pixel.Channel}, x={Pixel.X}, y={Pixel.Y} / RF {rf.Contributing}",18,h-35,12,Color.Wheat);
        Text("Dimensions: X=.45+.075 log2(W), Y=.45+.12 log2(H), Z=.4+.2 log2(C)",18,h-17,10,muted);
        ui.End();
        firstFrameMs ??= (DateTime.UtcNow-System.Diagnostics.Process.GetCurrentProcess().StartTime.ToUniversalTime()).TotalMilliseconds;
        if(proof is not null&&frame is 8 or 20 or 32 or 44 or 56 or 68)Capture(w,h);
        frame++;
        if(proof is not null&&frame>69){File.WriteAllText(Path.Combine(proof,"receipt.json"),JsonSerializer.Serialize(new{passed=true,firstFrameMs,frames=receipts},new JsonSerializerOptions{WriteIndented=true}));Exit();}
    }
    private void Capture(int w,int h)
    {
        // Read the actual rendered backbuffer, including FontStash text and the shared RF outline.
        var pixels=new Color[w*h];GraphicsDevice.GetBackBufferData(pixels);
        int nonBackground=pixels.Count(p=>p.R!=11||p.G!=20||p.B!=33);
        int text=0,yellow=0,stackPixels=0;
        var filterColors=new HashSet<uint>();
        for(int y=0;y<h;y++)for(int x=0;x<w;x++)
        {
            var p=pixels[y*w+x];
            if(x<Width&&y>60&&y<StackHeight-40&&(p.G>80||p.B>100))stackPixels++;
            var grid=GridRectangle;
            if(ShowFilters&&x>=grid.X&&x<grid.X+grid.Width&&y>=grid.Y&&y<grid.Y+grid.Height)filterColors.Add(p.PackedValue);
            if(pinned.Count>0&&x>=Width+18&&y>=244&&y<290&&p.R>150&&p.G>150&&p.B>150)text++;
            var r=PixelViews.PhotoRect(Width,Height);
            if(x>=r.X&&x<r.X+r.Width&&y>=r.Y&&y<r.Y+r.Height&&p.R>245&&p.G is >205 and <230&&p.B is >100 and <135)yellow++;
        }
        if(nonBackground<1000||stackPixels<500||(!ShowFilters&&yellow<4)||(pinned.Count>0&&text<30)||(ShowFilters&&filterColors.Count<32))throw new InvalidOperationException("Desktop pixel proof failed");
        string name=frame switch{8=>"default",20=>"hover-conv1a",32=>"pinned-card",44=>"selected-pixel",56=>"orbit",_=>"filters"};
        using var texture=new Texture2D(GraphicsDevice,w,h);texture.SetData(pixels);
        using(var stream=File.Create(Path.Combine(proof!,name+".png")))texture.SaveAsPng(stream,w,h);
        receipts.Add(new{name,nonBackground,stackPixels,filterColors=filterColors.Count,cardTextPixels=text,receptiveFieldPixels=yellow,hovered=Hovered,pinned=pinned.ToArray(),pixel=Pixel,camera=new{Camera.Yaw,Camera.Pitch,Camera.Zoom}});
    }
    protected override void Dispose(bool disposing){if(disposing){fonts?.Dispose();ui?.Dispose();white?.Dispose();}base.Dispose(disposing);}
}
