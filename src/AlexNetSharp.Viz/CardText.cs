using System.Text.RegularExpressions;

namespace AlexNetSharp.Viz;

public static class CardText
{
    public static string PlainBody(string body)=>Regex.Replace(body,@"\[([^\]]+)\]\(card:[^)]+\)","$1").Trim();
}
