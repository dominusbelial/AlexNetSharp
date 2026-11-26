using System.Diagnostics;
using AlexNetSharp.Replay;

var root = Path.GetFullPath(args.Length > 0 ? args[0] : ".");
var output = Path.GetFullPath(args.Length > 1 ? args[1] : "src/AlexNetSharp.Web/wwwroot/replay");
ulong seed = args.Length > 2 ? ulong.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 145;
var watch = Stopwatch.StartNew();
var snapshot = PlaceholderForward.Produce(root, seed);
snapshot.Write(output);
Console.WriteLine($"Snapshot: {snapshot.Manifest.Layers.Length} layers, {snapshot.Payload.Length + snapshot.ManifestBytes().Length} bytes, {watch.Elapsed.TotalSeconds:F3}s");
Console.WriteLine("Untrained placeholder weights; not ANX-11 initialisation. Preprocessing: not the ANX-12 pipeline.");
