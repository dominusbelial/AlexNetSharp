"""Reproduce VIZ-1 build/test/determinism gates with all writable state under scratch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
scratch = root / '.scratch/viz1'
scratch.mkdir(parents=True, exist_ok=True)
dotnet = Path.home() / '.dotnet/dotnet'
env = os.environ.copy()
for key, directory in {
    'HOME': 'tool-home', 'DOTNET_CLI_HOME': 'dotnet-home', 'NUGET_PACKAGES': 'nuget-packages',
    'NUGET_HTTP_CACHE_PATH': 'nuget-http-cache', 'TMPDIR': 'tmp', 'XDG_RUNTIME_DIR': 'runtime',
    'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data'
}.items():
    target = root / '.scratch' / directory
    target.mkdir(parents=True, exist_ok=True)
    env[key] = str(target)
env.update(DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_GENERATE_ASPNET_CERTIFICATE='false', DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE='true')
local = root / 'local.env'
if not env.get('ALEXNET_ORIGINAL_SRC') and local.exists():
    for line in local.read_text().splitlines():
        if line.startswith('ALEXNET_ORIGINAL_SRC='):
            env['ALEXNET_ORIGINAL_SRC'] = line.split('=', 1)[1].strip().strip('"\'')

results = []

def run(name, arguments):
    started = time.monotonic()
    with (scratch / (name + '.log')).open('w') as output:
        process = subprocess.Popen([str(dotnet), *arguments], cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
        code = process.wait()
    seconds = round(time.monotonic() - started, 3)
    result = dict(name=name, pid=process.pid, exit=code, seconds=seconds)
    results.append(result)
    (scratch / 'gates.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    assert code == 0, f'{name} failed; see scratch log'
    text = (scratch / (name + '.log')).read_text()
    if name.startswith('build'):
        assert '0 Warning(s)' in text and '0 Error(s)' in text
    if name.startswith('default'):
        assert seconds < 20, 'default process must remain under 20 s'

run('build-debug', ['build', '--disable-build-servers'])
run('build-release', ['build', '-c', 'Release', '--disable-build-servers'])
for number in range(1, 4):
    run(f'default-debug-{number}', ['tests/AlexNetSharp.Tests/bin/Debug/net10.0/AlexNetSharp.Tests.dll'])
run('full-release', ['tests/AlexNetSharp.Tests/bin/Release/net10.0/AlexNetSharp.Tests.dll', '--full-scalar-parity'])
run('snapshot-proof', ['tests/AlexNetSharp.Tests/bin/Release/net10.0/AlexNetSharp.Tests.dll', '--snapshot-proof'])
for number in range(1, 3):
    run(f'producer-{number}', ['tools/AlexNetSharp.Snapshot/bin/Release/net10.0/AlexNetSharp.Snapshot.dll', '.', f'.scratch/viz1/producer-{number}'])
checksums = {}
for filename in ['manifest.json', 'activations.bin']:
    a = (scratch / 'producer-1' / filename).read_bytes()
    b = (scratch / 'producer-2' / filename).read_bytes()
    committed = (root / 'src/AlexNetSharp.Web/wwwroot/replay' / filename).read_bytes()
    assert a == b == committed, 'fresh-process byte determinism: ' + filename
    checksums[filename] = dict(bytes=len(a), sha256=hashlib.sha256(a).hexdigest())
(scratch / 'determinism.json').write_text(json.dumps(checksums, indent=2) + '\n')
print('ALL BUILD/TEST/SNAPSHOT GATES PASSED', flush=True)
