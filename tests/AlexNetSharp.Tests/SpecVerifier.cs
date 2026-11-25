using System.Text.Json;

namespace AlexNetSharp.Tests;

internal static class SpecVerifier
{
    internal sealed record CfgSection(string Name, string Type, List<string> Inputs, string Neuron);

    public static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "AlexNetSharp.slnx")))
                return dir.FullName;
        }
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "AlexNetSharp.slnx")))
                return dir.FullName;
        }
        throw new InvalidOperationException("Could not locate repository root containing AlexNetSharp.slnx.");
    }

    public static string ResolveOriginalSrcRoot(string repoRoot)
    {
        var fromEnv = Environment.GetEnvironmentVariable("ALEXNET_ORIGINAL_SRC");
        if (!string.IsNullOrWhiteSpace(fromEnv))
        {
            if (!Directory.Exists(fromEnv))
                throw new InvalidOperationException($"ALEXNET_ORIGINAL_SRC directory does not exist: '{fromEnv}'.");
            return fromEnv;
        }

        var localEnvPath = Path.Combine(repoRoot, "local.env");
        if (File.Exists(localEnvPath))
        {
            foreach (var rawLine in File.ReadLines(localEnvPath))
            {
                var line = rawLine.Trim();
                if (line.Length == 0 || line.StartsWith('#'))
                    continue;
                if (line.StartsWith("ALEXNET_ORIGINAL_SRC=", StringComparison.Ordinal))
                {
                    var val = line["ALEXNET_ORIGINAL_SRC=".Length..].Trim().Trim('"', '\'');
                    if (Directory.Exists(val))
                        return val;
                    throw new InvalidOperationException($"ALEXNET_ORIGINAL_SRC from local.env does not exist: '{val}'.");
                }
            }
        }
        throw new InvalidOperationException("ALEXNET_ORIGINAL_SRC is not set in environment or local.env.");
    }

    public static List<CfgSection> ParseCfgSections(string cfgPath, List<string> errors)
    {
        if (!File.Exists(cfgPath))
        {
            errors.Add($"Missing original config file: {cfgPath}");
            return [];
        }

        var sections = new List<CfgSection>();
        string? currentName = null;
        string currentType = "";
        List<string> currentInputs = [];
        string currentNeuron = "";

        void Flush()
        {
            if (currentName is not null)
                sections.Add(new CfgSection(currentName, currentType, currentInputs, currentNeuron));
        }

        foreach (var raw in File.ReadLines(cfgPath))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
                continue;
            if (line.StartsWith('[') && line.EndsWith(']'))
            {
                Flush();
                currentName = line[1..^1].Trim();
                currentType = "";
                currentInputs = [];
                currentNeuron = "";
                continue;
            }
            var eq = line.IndexOf('=');
            if (eq <= 0 || currentName is null)
                continue;
            var key = line[..eq].Trim();
            var val = line[(eq + 1)..].Trim();
            if (key == "type")
                currentType = val;
            else if (key == "inputs")
                currentInputs = val.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
            else if (key == "neuron")
                currentNeuron = val;
        }
        Flush();
        return sections;
    }

    public static int VerifyAllCitations(
        JsonElement element,
        string origRoot,
        Dictionary<string, string[]> fileCache,
        string jsonPath,
        List<string> errors)
    {
        int verified = 0;
        if (element.ValueKind == JsonValueKind.Object)
        {
            bool hasPath = element.TryGetProperty("path", out var pProp);
            bool hasLine = element.TryGetProperty("line", out var lProp);
            bool hasToken = element.TryGetProperty("token", out var tProp);
            if (hasPath || hasLine || hasToken)
            {
                if (!(hasPath && hasLine && hasToken &&
                      pProp.ValueKind == JsonValueKind.String &&
                      lProp.ValueKind == JsonValueKind.Number && lProp.TryGetInt32(out int lineNo) &&
                      tProp.ValueKind == JsonValueKind.String))
                {
                    errors.Add($"{jsonPath}: malformed citation object (expected string 'path', int 'line', string 'token').");
                }
                else
                {
                    var relPath = pProp.GetString()!;
                    var token = tProp.GetString()!;
                    VerifySingleCitation(origRoot, fileCache, jsonPath, relPath, lineNo, token, errors);
                    verified++;
                }
            }
            foreach (var prop in element.EnumerateObject())
                verified += VerifyAllCitations(prop.Value, origRoot, fileCache, $"{jsonPath}.{prop.Name}", errors);
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            int idx = 0;
            foreach (var item in element.EnumerateArray())
            {
                verified += VerifyAllCitations(item, origRoot, fileCache, $"{jsonPath}[{idx}]", errors);
                idx++;
            }
        }
        return verified;
    }

    private static void VerifySingleCitation(
        string origRoot,
        Dictionary<string, string[]> fileCache,
        string jsonPath,
        string relPath,
        int lineNo,
        string token,
        List<string> errors)
    {
        if (string.IsNullOrWhiteSpace(relPath) || Path.IsPathRooted(relPath) || relPath.Contains("..", StringComparison.Ordinal))
        {
            errors.Add($"{jsonPath}: invalid relative citation path '{relPath}'.");
            return;
        }
        if (string.IsNullOrEmpty(token))
        {
            errors.Add($"{jsonPath} ({relPath}:{lineNo}): citation token must not be empty.");
            return;
        }
        if (!fileCache.TryGetValue(relPath, out var lines))
        {
            var fullPath = Path.Combine(origRoot, relPath.Replace('/', Path.DirectorySeparatorChar));
            if (!File.Exists(fullPath))
            {
                errors.Add($"{jsonPath} ({relPath}:{lineNo}): cited file '{relPath}' does not exist in original source tree.");
                return;
            }
            lines = File.ReadAllLines(fullPath);
            fileCache[relPath] = lines;
        }
        if (lineNo < 1 || lineNo > lines.Length)
        {
            errors.Add($"{jsonPath} ({relPath}:{lineNo}): cited line {lineNo} out of range 1..{lines.Length}.");
            return;
        }
        var actualLine = lines[lineNo - 1];
        if (!actualLine.Contains(token, StringComparison.Ordinal))
        {
            errors.Add($"{jsonPath} ({relPath}:{lineNo}): token \"{token}\" not found on line {lineNo} (actual: \"{actualLine.Trim()}\").");
        }
    }
}
