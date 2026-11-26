using System.Diagnostics;
using AlexNetSharp.Snapshot;

var root = Path.GetFullPath(args.Length > 0 ? args[0] : ".");
var output = Path.GetFullPath(args.Length > 1 ? args[1] : "src/AlexNetSharp.Web/wwwroot/replay");
ulong seed = args.Length > 2 && !args[2].StartsWith("--",StringComparison.Ordinal) ? ulong.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 145;
Directory.CreateDirectory(Path.Combine(root,".scratch"));
FileStream? generationLock=null;
for(int attempt=0;generationLock is null;attempt++)
{
    try{generationLock=new FileStream(Path.Combine(root,".scratch/replay-generation.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None);}
    catch(IOException) when(attempt<200){Thread.Sleep(50);}
}
using var heldLock=generationLock;
var watch = Stopwatch.StartNew();
var snapshot = PlaceholderForward.Produce(root, seed);
if (File.Exists(Path.Combine(output,"manifest.json")) && !args.Contains("--update-manifest") &&
    !snapshot.ManifestBytes().AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(output,"manifest.json"))))
    throw new InvalidDataException("Regeneration differs from the pinned manifest. Use --update-manifest only for an intentional snapshot change.");
snapshot.Write(output);
Console.WriteLine($"Snapshot: {snapshot.Manifest.Layers.Length} layers, {snapshot.Payload.Length + snapshot.ManifestBytes().Length} bytes, {watch.Elapsed.TotalSeconds:F3}s");
Console.WriteLine("Untrained placeholder weights; not ANX-11 initialisation. Preprocessing: not the ANX-12 pipeline.");
