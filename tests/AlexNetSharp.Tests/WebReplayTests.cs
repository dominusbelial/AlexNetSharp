using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using AlexNetSharp.Core.Config;
using AlexNetSharp.Replay;
using AlexNetSharp.Snapshot;
using AlexNetSharp.Viz;

namespace AlexNetSharp.Tests;
public static class WebReplayTests
{
    private static void Check(bool condition,string message){if(!condition)throw new Exception("VIZ-1: "+message);}
    private static ReplaySnapshot Load(string root)
    {
        string dir=Path.Combine(root,"src/AlexNetSharp.Web/wwwroot/replay"),manifest=Path.Combine(dir,"manifest.json"),payload=Path.Combine(dir,"activations.bin");
        if(!File.Exists(payload))
        {
            // activations.bin is generated, not tracked. Regenerate it and require the tracked manifest byte for byte.
            var produced=PlaceholderForward.Produce(root,145);
            Check(produced.ManifestBytes().AsSpan().SequenceEqual(File.ReadAllBytes(manifest)),"regenerated manifest equals the tracked manifest");
            File.WriteAllBytes(payload,produced.Payload.ToArray());
        }
        return ReplaySnapshot.Read(File.ReadAllBytes(manifest),File.ReadAllBytes(payload));
    }
    public static int RunAll(string root)
    {
        var watch=Stopwatch.StartNew();var snapshot=Load(root);var manifest=snapshot.Manifest;
        var layout=new LayerLayout(manifest);var camera=new OrbitCamera(layout.Span);int count=0;
        Check(layout.Boxes.Length==49&&layout.Boxes.Select(b=>b.Name).Distinct().Count()==49,"49 unique boxes");count++;
        foreach(bool portrait in new[]{false,true})
        {
            var fitted=new LayerLayout(manifest,portrait);var orbit=new OrbitCamera(fitted.Span);orbit.Fit(fitted.Span,portrait);
            int width=portrait?362:908,height=portrait?442:280;
            foreach(var box in fitted.Boxes)
            {
                var p=fitted.Project(box,orbit,width,height);
                Check(p.X>0&&p.X<width&&p.Y>0&&p.Y<height,"in fitted viewport "+box.Name+" portrait="+portrait+" point="+p);
                // Exact inverse projection ray, including nearest occluder in perspective.
                var picked=fitted.Pick(p.X,p.Y,orbit,width,height);
                Check(picked==box.Name,"exact projected ray "+box.Name+" hit="+picked+" portrait="+portrait);
                Check(fitted.PickRay(box.Center+Vector3.UnitZ*10,-Vector3.UnitZ)==box.Name,"known ray "+box.Name);
            }
            var conv=fitted.Boxes.Single(b=>b.Name=="conv1a");var point=fitted.Project(conv,orbit,width,height);
            Check(fitted.Pick(point.X,point.Y,orbit,width,height)=="conv1a","unoccluded conv1 center exact picking");
            var corners=fitted.Boxes.SelectMany(box=>from x in new[]{-1,1} from y in new[]{-1,1} from z in new[]{-1,1}
                select fitted.Project(box with{Center=box.Center+box.Size*new Vector3(x,y,z)/2},orbit,width,height)).ToArray();
            Check(corners.All(p=>p.X>=0&&p.X<=width&&p.Y>=0&&p.Y<=height),"all projected corners fit portrait="+portrait);
            Check(corners.Max(p=>p.X)-corners.Min(p=>p.X)>=width*.60f,"stack fills >=60% canvas width portrait="+portrait+" ratio="+((corners.Max(p=>p.X)-corners.Min(p=>p.X))/width));
            Check(orbit.Projection(width,height).M44==0,"perspective projection");
            foreach(var a in fitted.Boxes)foreach(var b in fitted.Boxes.Where(b=>string.CompareOrdinal(a.Name,b.Name)<0))
            {
                var d=Vector3.Abs(a.Center-b.Center);var half=(a.Size+b.Size)/2;
                Check(d.X>=half.X||d.Y>=half.Y||d.Z>=half.Z,"nonoverlap "+a.Name+" / "+b.Name);
            }
            orbit.Orbit(.08f,.04f);orbit.Magnify(.1f);
            point=fitted.Project(conv,orbit,width,height);
            Check(fitted.Pick(point.X,point.Y,orbit,width,height)=="conv1a","moved camera exact conv1 picking");count+=2;
        }
        Check(layout.Pick(-100,-100,camera,1280,280) is null,"miss outside scene");
        Check(layout.Edges.Length==manifest.Layers.Sum(l=>l.Inputs.Length),"all edges");
        foreach(string name in new[]{"conv3a","conv3b","fc2048a","fc2048b","fc2048ba","fc2048bb"})
            Check(layout.Edges.Any(e=>e.Consumer==name&&e.CrossTower),"distinct cross-tower "+name);
        Check(layout.Edges.Single(e=>e.Consumer=="fc2048bb"&&e.Slot==0).Input=="hs1b","reversed input slot 0");count++;
        float yaw=camera.Yaw,pitch=camera.Pitch;CameraControls.Key(camera,"ArrowRight");Check(camera.Yaw!=yaw,"shared keyboard orbit");
        camera.Orbit(100,-100);Check(camera.Yaw<=1.05f&&camera.Pitch>=-1.05f,"orbit limits");
        camera.Magnify(100);Check(camera.Zoom==4,"zoom upper limit");camera.Magnify(-100);Check(camera.Zoom==.6f,"zoom lower limit");
        camera.Reset();Check(camera.Yaw==yaw&&camera.Pitch==pitch&&camera.Zoom==1&&camera.Target==Vector3.Zero,"reset default three quarter angle");count++;
        var photo=Path.Combine(root,"content/samples/labrador.jpg");var provenance=Path.Combine(root,"content/samples/PROVENANCE.md");
        Check(File.Exists(provenance),"photo provenance exists");string provenanceText=File.ReadAllText(provenance);
        Check(Regex.IsMatch(provenanceText,@"Licence:.*(CC0|public domain)"),"explicit CC0 or PD licence line");
        Check(provenanceText.Contains(ReplaySnapshot.Hash(File.ReadAllBytes(photo))),"photo SHA-256 matches provenance");
        Check(new FileInfo(photo).Length<=200_000&&manifest.Image.Sha256==ReplaySnapshot.Hash(File.ReadAllBytes(photo)),"bounded photograph pinned by manifest");count++;
        var cases=new[]{
            ("conv1a",0,0,new PixelRange(0,0,10,10),new PixelRange(0,0,10,10)),
            ("conv1b",54,54,new PixelRange(216,216,223,223),new PixelRange(216,216,226,226)),
            ("pool1a",0,0,new PixelRange(0,0,18,18),new PixelRange(0,0,18,18)),
            ("pool1a",26,26,new PixelRange(208,208,223,223),new PixelRange(208,208,226,226)),
            ("conv2a",0,0,new PixelRange(0,0,34,34),new PixelRange(-16,-16,34,34)),
            ("conv2b",13,13,new PixelRange(88,88,138,138),new PixelRange(88,88,138,138)),
            ("pool3a",2,2,new PixelRange(0,0,210,210),new PixelRange(-16,-16,210,210)),
            ("pool3b",5,5,new PixelRange(80,80,223,223),new PixelRange(80,80,306,306))};
        var network=PlaceholderForward.Parse(root);
        // Independent 1D dependency enumeration: sets of actual indices, not RF intervals/recurrences.
        var enumerationCache=new Dictionary<(string,int),HashSet<int>>();
        HashSet<int> Enumerate(string name,int pixel)
        {
            if(enumerationCache.TryGetValue((name,pixel),out var cached))return cached;
            var layer=network[name];if(name=="data")return [pixel];
            var source=manifest.Layers.Single(l=>l.Name==layer.Inputs[0]);var indices=new HashSet<int>();
            int[] coordinates=layer.TypedParams switch
            {
                ConvLayerParams p=>Enumerable.Range(0,p.FilterSize[0]).Select(k=>pixel*p.Stride[0]-p.ConfigPadding[0]+k).ToArray(),
                PoolLayerParams p=>Enumerable.Range(0,p.SizeX).Select(k=>p.Start+pixel*p.Stride+k).ToArray(),
                RNormLayerParams p=>Enumerable.Range(0,p.Size).Select(k=>pixel-p.Size/2+k).ToArray(),
                _=>[pixel]
            };
            foreach(int i in coordinates.Where(i=>i>=0&&i<source.Shape[2]))indices.UnionWith(Enumerate(layer.Inputs[0],i));
            return enumerationCache[(name,pixel)]=indices;
        }
        foreach(var (name,x,y,expected,nominal) in cases)
        {
            var field=ReceptiveFields.Compute(manifest,name,x,y);
            Check(field.Contributing==expected&&field.Nominal==nominal,"hand receptive field "+name+" "+x);
            var independent=Enumerate(name,x);
            Check(independent.SetEquals(Enumerable.Range(expected.X0,expected.X1-expected.X0+1)),"independent receptive set "+name);count++;
        }
        var cards=Directory.GetFiles(Path.Combine(root,"content/explain"),"*.md").Select(p=>ExplainCard.Parse(File.ReadAllText(p))).ToDictionary(c=>c.Id);
        Check(cards.Count==59,"59 cards");
        Check(CardText.PlainBody("See [conv1](card:conv1a).")=="See conv1.","shared card link labels remain readable");
        Check(File.Exists(Path.Combine(root,"content/fonts/DejaVuSans.ttf"))&&File.ReadAllText(Path.Combine(root,"content/fonts/LICENSE.txt")).Contains("Permission is hereby granted"),"desktop font and permissive licence");
        var fontHash=ReplaySnapshot.Hash(File.ReadAllBytes(Path.Combine(root,"content/fonts/DejaVuSans.ttf")));
        Check(File.ReadAllText(Path.Combine(root,"content/fonts/LICENSE.txt")).Contains(fontHash),"bundled font checksum");count++;

        foreach(var l in manifest.Layers){Check(cards.ContainsKey(l.Name),"instance card "+l.Name);Check(cards.ContainsKey("type-"+l.Type),"type card "+l.Type);}
        foreach(var card in cards.Values)foreach(string link in card.Links)Check(cards.ContainsKey(link),"card link "+link);
        string? original=Environment.GetEnvironmentVariable("ALEXNET_ORIGINAL_SRC");
        int citations=0;
        if(string.IsNullOrWhiteSpace(original))Console.WriteLine("SKIP VIZ-1 original card citation resolution: ALEXNET_ORIGINAL_SRC is absent.");
        else foreach(var card in cards.Values)foreach(string citation in card.Citations)
        {
            if(citation.StartsWith("A:",StringComparison.Ordinal))continue;
            var m=Regex.Match(citation,@"^([^:]+):(\d+)$");Check(m.Success,"citation syntax "+citation);
            var path=Path.Combine(original,m.Groups[1].Value);Check(File.Exists(path),"citation file "+citation);
            Check(int.Parse(m.Groups[2].Value)>0&&File.ReadAllLines(path).Length>=int.Parse(m.Groups[2].Value),"citation line "+citation);citations++;
        }
        Check(!cards.Values.SelectMany(c=>c.Citations).Any(c=>c.StartsWith("A:")),"cards use original evidence only; no unresolved candidate citations");count+=2;
        var bytes=snapshot.ManifestBytes();
        Check(bytes.SequenceEqual(File.ReadAllBytes(Path.Combine(root,"src/AlexNetSharp.Web/wwwroot/replay/manifest.json"))),"canonical ordinal manifest round trip");var again=ReplaySnapshot.Read(bytes,snapshot.Payload.ToArray());
        Check(bytes.SequenceEqual(again.ManifestBytes()),"manifest round trip");Check(snapshot.Payload.SequenceEqual(again.Payload),"payload round trip");count++;
        void Reject(byte[] json,byte[] payload)
        {
            bool rejected=false;try{ReplaySnapshot.Read(json,payload);}catch(Exception e) when(e is InvalidDataException or JsonException){rejected=true;}
            Check(rejected,"invalid replay rejected");
        }
        Reject(Encoding.UTF8.GetBytes("{}"),snapshot.Payload);
        Reject(Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(bytes).Replace("\"schemaVersion\": 1","\"schemaVersion\": 999")),snapshot.Payload);
        var corrupt=snapshot.Payload.ToArray();corrupt[0]^=1;Reject(bytes,corrupt);
        Reject(bytes,snapshot.Payload[..^1]);Reject(bytes,new byte[ReplaySnapshot.MaximumBytes]);count++;
        Check(snapshot.Payload.Length+bytes.Length<10_000_000,"size bound");
        foreach(var l in manifest.Layers)
        {
            var values=snapshot.Values(l.Activations);
            if(values.Length>0){Check(values.Min()==l.Statistics.Min&&values.Max()==l.Statistics.Max,"statistics "+l.Name);}
        }
        Check(snapshot.Manifest.Top5.Select(c=>c.Index).SequenceEqual(snapshot.Values(manifest.Layers.Single(l=>l.Name=="probs").Activations).Select((p,i)=>(p,i)).OrderByDescending(v=>v.p).ThenBy(v=>v.i).Take(5).Select(v=>v.i)),"top 5");count++;
        var layer1=manifest.Layers.Single(l=>l.Name=="conv1a");var image=PixelViews.Activations(snapshot,layer1);var rect=PixelViews.Fit(image,PixelViews.GridRect(900,560));
        var pixel=PixelViews.Pick(rect.X+1.5f*rect.Width/image.Width,rect.Y+1.5f*rect.Height/image.Height,layer1,rect,image);
        Check(pixel==new ActivationPixel(0,0,0),"grid pixel pick");Check(PixelViews.Input(snapshot).Rgba.Length==224*224*4,"input texture");
        Check(PixelViews.Filters(snapshot).Width==208,"96 RGB filter atlas");count++;
        Console.WriteLine($"VIZ-1 TESTS PASSED: {count} groups; {citations} original card citations; {watch.Elapsed.TotalSeconds:F3}s.");return count;
    }
    public static int FullSnapshotProof(string root)
    {
        var watch=Stopwatch.StartNew();var first=PlaceholderForward.Produce(root);GC.Collect();var second=PlaceholderForward.Produce(root);
        Check(first.ManifestBytes().SequenceEqual(second.ManifestBytes()),"byte-identical manifests");
        Check(first.Payload.SequenceEqual(second.Payload),"byte-identical payloads");
        first.Write(Path.Combine(root,".scratch/viz1/test-replay"));
        var committed=Load(root);Check(first.ManifestBytes().SequenceEqual(committed.ManifestBytes())&&first.Payload.SequenceEqual(committed.Payload),"committed replay reproduces");
        var direct=PlaceholderForward.Run(PlaceholderForward.Parse(root),PlaceholderForward.LoadPhoto(root),145,2);
        foreach(var l in first.Manifest.Layers)
        {
            float[] expected=l.Activations.Count==0?[]:direct.Activations[l.Name].Storage.ToArray();
            Check(first.Values(l.Activations).Select(BitConverter.SingleToInt32Bits).SequenceEqual(expected.Select(BitConverter.SingleToInt32Bits)),"direct Core bits "+l.Name);
        }
        Check(first.Values(first.Manifest.Conv1Filters).SequenceEqual(direct.Conv1Filters),"direct filters");
        Check(first.Manifest.Cost.SequenceEqual(direct.Cost),"direct cost");
        Console.WriteLine($"SNAPSHOT FULL PROOF PASSED: two byte-identical productions, committed replay, 49 layer blobs equal direct Core forward at 2 workers, cost/filters; {first.ManifestBytes().Length+first.Payload.Length} bytes; {watch.Elapsed.TotalSeconds:F3}s.");return 0;
    }
}
