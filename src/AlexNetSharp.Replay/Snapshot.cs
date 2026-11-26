using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AlexNetSharp.Replay;

public sealed record ActivationStats(float Min, float Max, double Mean, double StandardDeviation, int Count);
public sealed record Blob(string File, int Offset, int Count, string Sha256);
public sealed record ReplayLayer(string Name, string Type, int[] Shape, int Tower, string[] Inputs,
    string[] Consumers, string? NeuronPlacement, long ParameterCount, SortedDictionary<string, string> Parameters,
    int Kernel, int Stride, int Start, ActivationStats Statistics, Blob Activations);
public sealed record RankedClass(int Index, float Probability);
public sealed record ImageInfo(string File, string Sha256, string Provenance, string License, string Preprocessing, float[] Mean);
public sealed record ReplayManifest(int SchemaVersion, long Step, string ConfigSha256, string ParamsSha256, ulong Seed,
    string Weights, string Dtype, string Layout, string Mode, ImageInfo Image, ReplayLayer[] Layers,
    Blob Conv1Filters, RankedClass[] Top5, double[] Cost, string[] Extensions);

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true, RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(ReplayManifest))]
public partial class ReplayJson : JsonSerializerContext;

public sealed class ReplaySnapshot(ReplayManifest manifest, byte[] payload)
{
    public const int MaximumBytes = 10_000_000;
    public ReplayManifest Manifest { get; } = manifest;
    public byte[] Payload { get; } = payload;
    public static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    public byte[] ManifestBytes() => JsonSerializer.SerializeToUtf8Bytes(Manifest, ReplayJson.Default.ReplayManifest);
    public static ReplaySnapshot Read(byte[] json, byte[] payload)
    {
        var manifest = JsonSerializer.Deserialize(json, ReplayJson.Default.ReplayManifest) ?? throw new InvalidDataException("Missing manifest.");
        // SortedDictionary deserialization otherwise uses the current culture comparer.
        if (manifest.Layers is null || manifest.Layers.Any(l => l.Parameters is null)) throw new InvalidDataException("Missing layers or parameters.");
        manifest = manifest with { Layers = manifest.Layers.Select(l => l with
            { Parameters = new SortedDictionary<string,string>(l.Parameters,StringComparer.Ordinal) }).ToArray() };
        var result = new ReplaySnapshot(manifest, payload);
        result.Validate(json.Length);
        return result;
    }
    public void Validate(int manifestBytes = 0)
    {
        if (Manifest.SchemaVersion != 1 || Manifest.Dtype != "float32-le" || Manifest.Layout != "channel,y,x" ||
            Manifest.Mode != "eval" || manifestBytes + Payload.Length >= MaximumBytes)
            throw new InvalidDataException("Unsupported or oversized replay.");
        var byName = Manifest.Layers.ToDictionary(l => l.Name, StringComparer.Ordinal);
        if (byName.Count == 0 || Manifest.Top5.Length != 5 || Manifest.ConfigSha256.Length != 64 || Manifest.ParamsSha256.Length != 64)
            throw new InvalidDataException("Invalid manifest header.");
        var blobs = Manifest.Layers.Select(l => l.Activations).Append(Manifest.Conv1Filters).ToArray();
        int offset = 0;
        foreach (var blob in blobs)
        {
            if (blob.File != "activations.bin" || blob.Offset != offset || blob.Count < 0 ||
                blob.Offset > Payload.Length - (long)blob.Count * 4) throw new InvalidDataException("Invalid blob range.");
            var bytes = Payload.AsSpan(blob.Offset, blob.Count * 4);
            if (Hash(bytes) != blob.Sha256) throw new InvalidDataException("Blob checksum mismatch.");
            offset += blob.Count * 4;
            foreach (float v in Values(blob)) if (!float.IsFinite(v)) throw new InvalidDataException("Nonfinite activation.");
        }
        if (offset != Payload.Length || Manifest.Conv1Filters.Count != 96 * 3 * 11 * 11) throw new InvalidDataException("Invalid payload size.");
        foreach (var l in byName.Values)
        {
            if (l.Shape.Length != 3 || l.Shape.Any(n => n < 0) || (long)l.Shape[0] * l.Shape[1] * l.Shape[2] != l.Activations.Count ||
                l.Statistics.Count != l.Activations.Count || l.Stride < 1 || l.Kernel < 1 || l.ParameterCount < 0)
                throw new InvalidDataException("Invalid layer dimensions.");
            foreach (string input in l.Inputs) if (!byName.TryGetValue(input, out var p) || !p.Consumers.Contains(l.Name)) throw new InvalidDataException("Broken graph.");
            foreach (string consumer in l.Consumers) if (!byName.TryGetValue(consumer, out var c) || !c.Inputs.Contains(l.Name)) throw new InvalidDataException("Broken consumer.");
        }
        var visited = new HashSet<string>(); var active = new HashSet<string>();
        void Visit(string name)
        {
            if (visited.Contains(name)) return;
            if (!active.Add(name)) throw new InvalidDataException("Cycle in graph.");
            foreach (var input in byName[name].Inputs) Visit(input);
            active.Remove(name); visited.Add(name);
        }
        foreach (var name in byName.Keys) Visit(name);
    }
    public float[] Values(Blob blob)
    {
        var result = new float[blob.Count];
        for (int i = 0; i < result.Length; i++) result[i] = BinaryPrimitives.ReadSingleLittleEndian(Payload.AsSpan(blob.Offset + i * 4, 4));
        return result;
    }
    public void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        var json = ManifestBytes(); Validate(json.Length);
        File.WriteAllBytes(Path.Combine(directory, "manifest.json"), json);
        File.WriteAllBytes(Path.Combine(directory, "activations.bin"), Payload);
    }
}

// A future live source can implement the same contract. Each step is immutable.
public interface ISnapshotSource { Task<ReplaySnapshot> LoadAsync(CancellationToken cancellationToken = default); }
// Deliberately empty: audio has no implementation in this slice.
public interface IAudioLayer;
