using System.Globalization;
using AlexNetSharp.Replay;
using AlexNetSharp.Viz;
using Microsoft.JSInterop;

namespace AlexNetSharp.Web.Pages;
public partial class Index
{
    private ReplaySnapshot? _snapshot;
    private StackGame? _game;
    private DotNetObjectReference<Index>? _reference;
    private string? _error,_hover;
    private readonly Dictionary<string,ExplainCard> _cards=new();
    private readonly List<string> _pinned=[];
    private ScreenPoint[] _points=[];
    private ReceptiveField? _field;
    private static string F(float n)=>n.ToString("F2",CultureInfo.InvariantCulture);
    private static string G(double n)=>n.ToString("G5",CultureInfo.InvariantCulture);
    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if(!firstRender) return;
        try
        {
            _snapshot=await new ReplaySource(Http).LoadAsync();
            var ids=_snapshot.Manifest.Layers.Select(l=>l.Name).Concat(_snapshot.Manifest.Layers.Select(l=>"type-"+l.Type)).Distinct();
            await Task.WhenAll(ids.Select(async id=>
            {
                var card=ExplainCard.Parse(await Http.GetStringAsync($"explain/{id}.md")); _cards.Add(id,card);
            }));
            _game=new(_snapshot); _reference=DotNetObjectReference.Create(this);
            UpdateField(); StateHasChanged(); await JS.InvokeVoidAsync("viz.start",_reference);
        }
        catch(Exception e){_error="The replay could not start: "+e.Message;StateHasChanged();}
    }
    [JSInvokable] public bool Frame(int width,int height)
    {
        try
        {
            if(_game is null) return false;
            _game.Size(width,height);
            if(!_game.Started){_game.Run();_game.Started=true;_game.Size(width,height);RefreshPoints();}
            _game.Tick();return true;
        }
        catch(Exception e){_error="Renderer stopped: "+e.Message;StateHasChanged();return false;}
    }
    private void RefreshPoints()
    {
        if(_game is null) return;
        _points=_game.Layout.Boxes.Select(b=>_game.Layout.Project(b,_game.Camera,_game.Width,_game.StackHeight)).ToArray(); StateHasChanged();
    }
    [JSInvokable] public void Resize(){RefreshPoints();}
    [JSInvokable] public void Pointer(float x,float y,bool click)
    {
        if(_game is null) return;
        string? name=y<_game.StackHeight?_game.Layout.Pick(x,y,_game.Camera,_game.Width,_game.StackHeight):null;
        _hover=name;_game.Hovered=name;
        if(click && name is not null) Pin(name);
        if(click&&!_game.ShowFilters)
        {
            var pixel=PixelViews.Pick(x,y,_game.Layer,_game.GridRectangle,_game.GridImage);
            if(pixel is not null){_game.Pixel=pixel;UpdateField();}
        }
        StateHasChanged();
    }
    [JSInvokable] public void MoveCamera(float dx,float dy,bool pan)
    {
        if(_game is null)return;
        if(pan)_game.Camera.Pan(-dx*.06f,dy*.06f);else _game.Camera.Orbit(dx*.006f,dy*.006f);
        RefreshPoints();
    }
    [JSInvokable] public void Wheel(float delta){_game?.Camera.Magnify(-delta*.001f);RefreshPoints();}
    [JSInvokable] public void Key(string key)
    {
        if(_game is null)return;
        CameraControls.Key(_game.Camera,key);
        RefreshPoints();
    }
    private void CameraAction(string action){if(action=="reset")Key("r");else Key(action switch{"left"=>"ArrowLeft","right"=>"ArrowRight","in"=>"+",_=>"-"});}
    private void Pin(string id)
    {
        if(!_cards.ContainsKey(id))return;
        if(!_pinned.Contains(id))_pinned.Add(id);
        _game?.Select(id);UpdateField();StateHasChanged();
    }
    private void Filters(bool show){if(_game is not null)_game.ShowFilters=show;StateHasChanged();}
    private void SetPixel(string axis,object? value)
    {
        if(_game is null||!int.TryParse(value?.ToString(),out int n))return;
        var p=_game.Pixel;_game.Pixel=axis switch
        {
            "c"=>p with{Channel=Math.Clamp(n,0,_game.Layer.Shape[0]-1)},
            "x"=>p with{X=Math.Clamp(n,0,_game.Layer.Shape[2]-1)},
            _=>p with{Y=Math.Clamp(n,0,_game.Layer.Shape[1]-1)}
        };UpdateField();
    }
    private void UpdateField(){if(_game is not null&&_snapshot is not null)_field=ReceptiveFields.Compute(_snapshot.Manifest,_game.Selected,_game.Pixel.X,_game.Pixel.Y);}
    private float FirstValue(ReplayLayer layer)=>layer.Activations.Count==0?0:_snapshot!.Values(layer.Activations)[0];
    private float SelectedValue()
    {
        if(_snapshot is null||_game is null)return 0;
        var p=_game.Pixel;int side=_game.Layer.Shape[1];
        return _snapshot.Values(_game.Layer.Activations)[(p.Channel*side+p.Y)*side+p.X];
    }
    private string LiveFormula(ReplayLayer layer)
    {
        string Param(string key)=>layer.Parameters.GetValueOrDefault(key,"—");
        float Input0()=>_snapshot!.Values(_snapshot.Manifest.Layers.Single(l=>l.Name==layer.Inputs[0]).Activations)[0];
        return layer.Type switch
        {
            "cmrnorm"=>$"denom = {Param("minDiv")} + {Param("scale")} Σ x²; y = x × denom^(-{Param("pow")}); size = {Param("size")}",
            "rnorm"=>$"denom = 1 + {Param("scale")} Σ x²; y = x × denom^(-{Param("pow")}); size = {Param("size")}",
            "hs"=>$"y[0] = {Param("keep")} × {G(Input0())} = {G(FirstValue(layer))} (eval)",
            "neuron"=>$"y[0] = max(0, {G(Input0())}) = {G(FirstValue(layer))}",
            "conv"=>$"y[0] = initB({Param("initB")}) + weighted sum = {G(FirstValue(layer))}; window {layer.Kernel}×{layer.Kernel}, stride {layer.Stride}, start {layer.Start}",
            "pool"=>$"y[0] = max of the clipped {layer.Kernel}×{layer.Kernel} window = {G(FirstValue(layer))}",
            "softmax"=>$"p[0] = {G(FirstValue(layer))}; Σ p = {_snapshot!.Values(layer.Activations).Sum(v=>(double)v).ToString("G6",CultureInfo.InvariantCulture)}",
            "cost.logreg"=>$"label = 0; -log(p[0]) = {G(_snapshot!.Manifest.Cost[0])}; top-1 error = {G(_snapshot.Manifest.Cost[1])}; top-5 error = {G(_snapshot.Manifest.Cost[2])}",
            "fc"=>$"ordered inputs: {string.Join(" → ",layer.Inputs)}; bias = {Param("initB")}; y[0] = {G(FirstValue(layer))}",
            _=>layer.Name=="data"?$"x[0] = RGB[0] - 123 = {G(FirstValue(layer))}":"Synthetic label index = 0"
        };
    }
    private static string PlainBody(string body)=>CardText.PlainBody(body);
    // Proof hooks return coordinates computed by the very same pure geometry used to draw and pick.
    [JSInvokable] public ScreenPoint[] LayoutPoints()=>_points;
    [JSInvokable] public object ViewState()=>new{selected=_game?.Selected,grid=_game?.GridRectangle,field=_field,pixel=_game?.Pixel,frames=_game?.Started,camera=_game is null?null:new{_game.Camera.Yaw,_game.Camera.Pitch,_game.Camera.Zoom},stackBounds=StackBounds()};
    private object? StackBounds()
    {
        if(_game is null)return null;
        var points=_game.Layout.Boxes.SelectMany(box=>from x in new[]{-1,1} from y in new[]{-1,1} from z in new[]{-1,1}
            select _game.Layout.Project(box with{Center=box.Center+box.Size*new System.Numerics.Vector3(x,y,z)/2},_game.Camera,_game.Width,_game.StackHeight)).ToArray();
        return new{x=points.Min(p=>p.X),y=points.Min(p=>p.Y),width=points.Max(p=>p.X)-points.Min(p=>p.X),height=points.Max(p=>p.Y)-points.Min(p=>p.Y)};
    }
    public void Dispose(){_game?.Dispose();_reference?.Dispose();}
}
