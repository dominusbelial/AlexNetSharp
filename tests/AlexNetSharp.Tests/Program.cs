using AlexNetSharp.Tests;

try
{
    var repoRoot = SpecVerifier.FindRepoRoot();
    var origRoot = SpecVerifier.ResolveOriginalSrcRoot(repoRoot);

    string ledgerPath = Path.Combine(repoRoot, "docs", "spec", "ledger.json");
    string manifestPath = Path.Combine(repoRoot, "docs", "spec", "experiment.json");
    bool ledgerOnly = false;

    for (int i = 0; i < args.Length; i++)
    {
        if (args[i] == "--ledger" && i + 1 < args.Length)
            ledgerPath = Path.GetFullPath(args[++i]);
        else if (args[i] == "--manifest" && i + 1 < args.Length)
            manifestPath = Path.GetFullPath(args[++i]);
        else if (args[i] == "--ledger-only")
            ledgerOnly = true;
    }

    var errors = new List<string>();
    var fileCache = new Dictionary<string, string[]>(StringComparer.Ordinal);
    var cfgPath = Path.Combine(origRoot, "layers", "layers-145.cfg");
    var cfgSections = SpecVerifier.ParseCfgSections(cfgPath, errors);

    var (secCount, typeCount, ledgerCitations) =
        LedgerAndManifestChecks.VerifyLedger(origRoot, ledgerPath, fileCache, cfgSections, errors);

    int weightedCount = 0;
    int manifestCitations = 0;
    if (!ledgerOnly)
    {
        (weightedCount, manifestCitations) =
            LedgerAndManifestChecks.VerifyManifest(origRoot, manifestPath, fileCache, cfgSections, errors);
    }

    if (errors.Count > 0)
    {
        Console.Error.WriteLine($"SPEC CHECK FAILED ({errors.Count} error(s)):");
        foreach (var err in errors)
            Console.Error.WriteLine($"  - {err}");
        return 1;
    }

    Console.WriteLine(ledgerOnly
        ? $"SPEC CHECK PASSED (ledger-only): {secCount} sections, {typeCount} layer types, {ledgerCitations} citations verified."
        : $"SPEC CHECK PASSED: {secCount} sections, {typeCount} layer types, {weightedCount} weighted layers, {ledgerCitations + manifestCitations} total citations verified.");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"SPEC CHECK ERROR: {ex.Message}");
    return 1;
}
