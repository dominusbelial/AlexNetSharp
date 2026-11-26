namespace AlexNetSharp.Replay;

public sealed record ExplainCard(string Id,string Title,string Summary,string PaperSection,string[] Citations,string Body,string Formula,string[] Links)
{
    public static ExplainCard Parse(string markdown)
    {
        string[] lines = markdown.Replace("\r", "").Split('\n');
        if (lines[0] != "---") throw new InvalidDataException("Card front matter missing.");
        int end = Array.IndexOf(lines,"---",1);
        if (end < 0) throw new InvalidDataException("Card front matter not closed.");
        var fields = lines[1..end].Select(l => l.Split(':',2)).ToDictionary(p => p[0],p => p[1].Trim());
        foreach (string key in new[]{"id","title","summary","paper-section","citations"})
            if (!fields.ContainsKey(key) || fields[key].Length == 0) throw new InvalidDataException($"Card missing {key}.");
        string body = string.Join('\n',lines[(end+1)..]);
        int formula = body.IndexOf("Formula:",StringComparison.Ordinal);
        if (formula < 0 || !body.Contains("Why it matters:")) throw new InvalidDataException("Card needs explanation and formula.");
        var links = System.Text.RegularExpressions.Regex.Matches(body,@"\]\(card:([\w.-]+)\)").Select(m => m.Groups[1].Value).ToArray();
        return new(fields["id"],fields["title"],fields["summary"],fields["paper-section"],fields["citations"].Split(';',StringSplitOptions.TrimEntries),
            body[..formula].Trim(),body[(formula+8)..].Trim(),links);
    }
}
