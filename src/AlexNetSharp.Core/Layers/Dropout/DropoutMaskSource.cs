using AlexNetSharp.Core.Tensors;

namespace AlexNetSharp.Core.Layers.Dropout;

/// <summary>Writes one binary mask value per (feature,case). Called once per enabled
/// training forward, never by backward or evaluation. Sources belong to one layer/lane.</summary>
public interface IDropoutMaskSource
{
    void Fill(FloatMatrix mask, float keep, int maxDegreeOfParallelism = 1);
}

/// <summary>Repeatable injected mask in canonical feature-major, case-fastest order.</summary>
public sealed class FixedDropoutMaskSource : IDropoutMaskSource
{
    private readonly float[] _values;
    public FixedDropoutMaskSource(IEnumerable<float> values)
    {
        _values = values.ToArray();
        if (_values.Any(v => v != 0 && v != 1)) throw new ArgumentException("Mask values must be binary.");
    }
    public void Fill(FloatMatrix mask, float keep, int maxDegreeOfParallelism = 1)
    {
        if (mask.ElementCount != _values.Length) throw new ArgumentException("Fixed mask dimensions disagree.");
        for (int f = 0; f < mask.Rows; f++) for (int n = 0; n < mask.Cols; n++)
            mask[f, n] = _values[f * mask.Cols + n];
    }
}

/// <summary>Port RNG, NOT the historical CUDA stream. SplitMix64 hashes
/// seed + 0x9E3779B97F4A7C15*(drawOffset+feature*cases+case+1), modulo 2^64.
/// u=(top24bits+1)*2^-24 is in (0,1]; mask=(u&lt;keep), strictly.
/// DrawOffset advances by the element count after each fill; retain seed/offset
/// for future checkpointing. Independent of worker count; not concurrently callable.</summary>
public sealed class SeededDropoutMaskSource(ulong seed, ulong drawOffset = 0) : IDropoutMaskSource
{
    public ulong Seed { get; } = seed;
    public ulong DrawOffset { get; private set; } = drawOffset;
    public static float Uniform(ulong seed, ulong index)
    {
        ulong z = unchecked(seed + 0x9E3779B97F4A7C15UL * (index + 1));
        z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
        z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
        z ^= z >> 31;
        return ((z >> 40) + 1) * (1f / 16777216f);
    }
    public void Fill(FloatMatrix mask, float keep, int maxDegreeOfParallelism = 1)
    {
        if (maxDegreeOfParallelism == 0 || maxDegreeOfParallelism < -1) throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism));
        ulong start = DrawOffset;
        Parallel.For(0, mask.Cols, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, n =>
        {
            for (int f = 0; f < mask.Rows; f++)
                mask[f, n] = Uniform(Seed, unchecked(start + (ulong)f * (ulong)mask.Cols + (ulong)n)) < keep ? 1 : 0;
        });
        DrawOffset = unchecked(start + (ulong)mask.ElementCount);
    }
}
