using AlexNetSharp.Core.Config;

namespace AlexNetSharp.Tests;

internal sealed class TestFailureException(string message) : Exception(message);

internal static class TestAssert
{
    public static void True(bool condition, string message)
    {
        if (!condition)
            throw new TestFailureException(message);
    }

    public static void Equal<T>(T expected, T actual, string label)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            throw new TestFailureException($"{label}: expected '{expected}', got '{actual}'.");
    }

    public static void FloatEqual(double expected, double actual, string label, double tol = 1e-15)
    {
        if (Math.Abs(expected - actual) > tol)
            throw new TestFailureException($"{label}: expected {expected:G17}, got {actual:G17}.");
    }

    public static void SequenceEqual<T>(IReadOnlyList<T> expected, IReadOnlyList<T> actual, string label)
    {
        if (!expected.SequenceEqual(actual))
        {
            throw new TestFailureException(
                $"{label}: expected [{string.Join(", ", expected)}], got [{string.Join(", ", actual)}].");
        }
    }

    public static LayerParsingException ThrowsLayerError(Action action, string expectedSubstring, string testLabel)
    {
        try
        {
            action();
        }
        catch (LayerParsingException ex)
        {
            if (!ex.Message.Contains(expectedSubstring, StringComparison.Ordinal))
            {
                throw new TestFailureException(
                    $"{testLabel}: expected message containing \"{expectedSubstring}\", but got \"{ex.Message}\".");
            }
            return ex;
        }
        throw new TestFailureException(
            $"{testLabel}: expected LayerParsingException containing \"{expectedSubstring}\", but no exception was thrown.");
    }
}
