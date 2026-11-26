using AlexNetSharp.Web;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

Microsoft.Xna.Platform.GameFactory.RegisterGameFactory(new Microsoft.Xna.Platform.ConcreteGameFactory());
Microsoft.Xna.Platform.Input.InputFactory.RegisterInputFactory(new Microsoft.Xna.Platform.Input.ConcreteInputFactory());
Microsoft.Xna.Platform.Graphics.GraphicsFactory.RegisterGraphicsFactory(new Microsoft.Xna.Platform.Graphics.ConcreteGraphicsFactory());
var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");
builder.Services.AddScoped(_ => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
await builder.Build().RunAsync();
