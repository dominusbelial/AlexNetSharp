using AlexNetSharp.Replay;
namespace AlexNetSharp.Web;
public sealed class ReplaySource(HttpClient http) : ISnapshotSource
{
    public async Task<ReplaySnapshot> LoadAsync(CancellationToken cancellationToken = default)
    {
        var json = await http.GetByteArrayAsync("replay/manifest.json",cancellationToken);
        var bytes = await http.GetByteArrayAsync("replay/activations.bin",cancellationToken);
        return ReplaySnapshot.Read(json,bytes);
    }
}
