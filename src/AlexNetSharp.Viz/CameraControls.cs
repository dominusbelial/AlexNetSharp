namespace AlexNetSharp.Viz;

public static class CameraControls
{
    public static void Key(OrbitCamera camera,string key)
    {
        switch(key.ToLowerInvariant())
        {
            case "left":case "arrowleft":camera.Orbit(-.04f,0);break;
            case "right":case "arrowright":camera.Orbit(.04f,0);break;
            case "up":case "arrowup":camera.Orbit(0,-.04f);break;
            case "down":case "arrowdown":camera.Orbit(0,.04f);break;
            case "a":camera.Pan(-.5f,0);break;case "d":camera.Pan(.5f,0);break;
            case "w":camera.Pan(0,.5f);break;case "s":camera.Pan(0,-.5f);break;
            case "+":case "=":camera.Magnify(.12f);break;case "-":camera.Magnify(-.12f);break;
            case "r":camera.Reset();break;
        }
    }
}
