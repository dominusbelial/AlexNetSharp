"""Opt-in SDL2/GL proof. Clean regeneration, offscreen first, bounded display fallback."""
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[1]
scratch = root / '.scratch/viz2'
scratch.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
for key, directory in {
    'DOTNET_CLI_HOME': 'dotnet-home', 'NUGET_PACKAGES': 'nuget-packages',
    'NUGET_HTTP_CACHE_PATH': 'nuget-http-cache', 'TMPDIR': 'tmp', 'XDG_RUNTIME_DIR': 'runtime',
    'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data'
}.items():
    target = root / '.scratch' / directory
    target.mkdir(parents=True, exist_ok=True)
    env[key] = str(target)
env.update(DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_GENERATE_ASPNET_CERTIFICATE='false', DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE='true')
dotnet = Path.home() / '.dotnet/dotnet'
# Each proof starts without the generated payload or Web build intermediates.
(root / 'src/AlexNetSharp.Web/wwwroot/replay/activations.bin').unlink(missing_ok=True)
for directory in ['src/AlexNetSharp.Web/obj', 'src/AlexNetSharp.Web/bin']:
    shutil.rmtree(root / directory, ignore_errors=True)
with (scratch / 'desktop-build.log').open('w') as log:
    subprocess.run([str(dotnet), 'build', 'src/AlexNetSharp.Desktop', '-c', 'Release', '--disable-build-servers'], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
build = (scratch / 'desktop-build.log').read_text()
assert '0 Warning(s)' in build and '0 Error(s)' in build
out = scratch / 'desktop'
shutil.rmtree(out, ignore_errors=True)
attempts = []

def no_core_dump():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

for mode in ['offscreen', 'display']:
    current_env = env.copy()
    if mode == 'offscreen':
        current_env['SDL_VIDEODRIVER'] = 'offscreen'
    else:
        current_env.pop('SDL_VIDEODRIVER', None)
    started = time.monotonic()
    with (scratch / ('desktop-' + mode + '.log')).open('w') as log:
        process = subprocess.Popen([str(dotnet), 'src/AlexNetSharp.Desktop/bin/Release/net10.0/AlexNetSharp.Desktop.dll', '--proof', str(out)], cwd=root, env=current_env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=no_core_dump)
        try:
            code = process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            process.kill()  # Only this driver's own child PID.
            code = process.wait()
    attempts.append(dict(mode=mode, exit=code, pid=process.pid, seconds=round(time.monotonic()-started, 3)))
    if code == 0:
        break
assert attempts[-1]['exit'] == 0, 'Desktop proof failed; see SDL logs'
receipt = json.loads((out / 'receipt.json').read_text())
frames = {frame['name']: frame for frame in receipt['frames']}
assert receipt['passed'] and set(frames) == {'default', 'hover-conv1a', 'pinned-card', 'selected-pixel', 'orbit', 'filters'}
assert frames['hover-conv1a']['hovered'] == 'conv1a'
assert frames['pinned-card']['pinned'] == ['conv1a']
assert frames['selected-pixel']['pixel'] == {'Channel': 0, 'X': 3, 'Y': 4}
assert frames['default']['camera'] != frames['orbit']['camera']
for name, frame in frames.items():
    assert frame['nonBackground'] > 1000 and frame['stackPixels'] > 500
    if frame['pinned']:
        assert frame['cardTextPixels'] >= 30
    if name != 'filters':
        assert frame['receptiveFieldPixels'] >= 4
    assert (out / (name + '.png')).stat().st_size > 1000
assert frames['filters']['filterColors'] > 32
(scratch / 'desktop-proof.json').write_text(json.dumps(dict(passed=True, attempts=attempts, receipt=receipt), indent=2) + '\n')
print(json.dumps(dict(passed=True, attempts=attempts), indent=2))
