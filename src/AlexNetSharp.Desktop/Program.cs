using AlexNetSharp.Desktop;
using AlexNetSharp.Replay;

Microsoft.Xna.Platform.GameFactory.RegisterGameFactory(new Microsoft.Xna.Platform.ConcreteGameFactory());
Microsoft.Xna.Platform.Input.InputFactory.RegisterInputFactory(new Microsoft.Xna.Platform.Input.ConcreteInputFactory());
Microsoft.Xna.Platform.Graphics.GraphicsFactory.RegisterGraphicsFactory(new Microsoft.Xna.Platform.Graphics.ConcreteGraphicsFactory());
string? proof=null,path=null;
for(int i=0;i<args.Length;i++)
{
    if(args[i]=="--proof" && i+1<args.Length)proof=Path.GetFullPath(args[++i]);
    else if(path is null)path=Path.GetFullPath(args[i]);
    else throw new ArgumentException("Usage: AlexNetSharp.Desktop [replay-directory] [--proof output-directory]");
}
if(path is null)
{
    var root=new DirectoryInfo(AppContext.BaseDirectory);
    while(root is not null&&!File.Exists(Path.Combine(root.FullName,"AlexNetSharp.slnx")))root=root.Parent;
    path=root is null?Path.Combine(AppContext.BaseDirectory,"replay"):Path.Combine(root.FullName,"src/AlexNetSharp.Web/wwwroot/replay");
}
var snapshot=ReplaySnapshot.Read(File.ReadAllBytes(Path.Combine(path,"manifest.json")),File.ReadAllBytes(Path.Combine(path,"activations.bin")));
using var game=new DesktopGame(snapshot,proof);
game.Run();
