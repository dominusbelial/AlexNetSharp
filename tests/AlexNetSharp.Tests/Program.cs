using System.Diagnostics;
using System.Text.Json.Nodes;
using AlexNetSharp.Tests;

try
{
    var sw = Stopwatch.StartNew();
    var repoRoot = SpecVerifier.FindRepoRoot();
    var origRoot = SpecVerifier.ResolveOriginalSrcRoot(repoRoot);

    string ledgerPath = Path.Combine(repoRoot, "docs", "spec", "ledger.json");
    string manifestPath = Path.Combine(repoRoot, "docs", "spec", "experiment.json");
    string goldenPath = Path.Combine(repoRoot, "tests", "fixtures", "layers-145.parsed.json");
    bool ledgerOnly = false;
    bool specOnly = false;
    bool writeGolden = false;

    for (int i = 0; i < args.Length; i++)
    {
        if (args[i] == "--ledger" && i + 1 < args.Length)
            ledgerPath = Path.GetFullPath(args[++i]);
        else if (args[i] == "--manifest" && i + 1 < args.Length)
            manifestPath = Path.GetFullPath(args[++i]);
        else if (args[i] == "--ledger-only")
            ledgerOnly = true;
        else if (args[i] == "--spec-only")
            specOnly = true;
        else if (args[i] == "--write-golden")
            writeGolden = true;
    }

    if (writeGolden)
    {
        GoldenParseTests.WriteGoldenFixture(origRoot, goldenPath);
        Console.WriteLine($"WROTE GOLDEN FIXTURE: {goldenPath}");
        return 0;
    }

    var errors = new List<string>();
    var fileCache = new Dictionary<string, string[]>(StringComparer.Ordinal);
    var cfgPath = Path.Combine(origRoot, "layers", "layers-145.cfg");
    var cfgSections = SpecVerifier.ParseCfgSections(cfgPath, errors);

    var (secCount, typeCount, ledgerCitations) =
        LedgerAndManifestChecks.VerifyLedger(origRoot, ledgerPath, fileCache, cfgSections, errors);

    int linkedParams = SpecLinkChecks.VerifyParserAgainstLedger(origRoot, ledgerPath, errors);
    int linkedShapes = ShapeSpecLinkChecks.Verify(ShapeTests.Parse145(origRoot),
        JsonNode.Parse(File.ReadAllText(ledgerPath))!, errors);

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
        ? $"SPEC CHECK PASSED (ledger-only): {secCount} sections, {typeCount} layer types, {linkedParams} parser params linked, {ledgerCitations} citations verified."
        : $"SPEC CHECK PASSED: {secCount} sections, {typeCount} layer types, {weightedCount} weighted layers, {linkedParams} parser params linked, {ledgerCitations + manifestCitations} total citations verified.");
    Console.WriteLine($"SHAPE SPEC LINK PASSED: {linkedShapes} fields, 49 layers at each of 224 and 227.");

    if (ledgerOnly || specOnly)
        return 0;

    int goldenPassed = GoldenParseTests.RunAll(repoRoot, origRoot);
    int errorPassed = ConfigParserErrorTests.RunAll();
    int shapesPassed = ShapeTests.RunAll(repoRoot, origRoot);
    sw.Stop();

    Console.WriteLine(
        $"CONFIG PARSER TESTS PASSED: {goldenPassed} golden/smoke tests + {errorPassed} error/INI rule tests ({goldenPassed + errorPassed} total) in {sw.ElapsedMilliseconds} ms.");
    Console.WriteLine($"SHAPE TESTS PASSED: {shapesPassed} full-network, hand, boundary/error and ledger enforcement tests.");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"TEST SUITE FAILED: {ex.Message}");
    return 1;
}
