"""Disposable Docker execution with per-candidate mounts and shared render-GPU locks."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import re
import time
import uuid

IMAGE = 'vllm/vllm-openai:v0.19.0'
RUNTIME = Path(os.environ.get('BG_AGENT_RUNTIME', '/limx_embap/tos/user/Yaleon/projects/blender-research/eval-agentic-20261015/sandbox-runtime'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def build_command(task, script, output, blender, infinigen, gpu, name, samples=512, seed=20261015):
    # Host control files never live in the mounted output directory.
    script, output = Path(script).resolve(), Path(output).resolve()
    blender, infinigen = Path(blender).resolve(), Path(infinigen).resolve()
    base = Path(task['base_blend']).resolve()
    driver_dir = RUNTIME / 'drivers'
    runner_dir = RUNTIME / 'runner'
    kernel_dir = RUNTIME / 'kernels'
    base_dir = RUNTIME / 'bases' / base.stem
    input_dir = output.parent / 'input'
    for directory in [driver_dir, runner_dir, kernel_dir, base_dir, input_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    # This host's Docker volume adapter supports directory binds only.
    # Stage a minimal public runtime and a single base scene, never a dataset folder.
    with (RUNTIME / 'stage.lock').open('a') as stage_lock:
        fcntl.flock(stage_lock, fcntl.LOCK_EX)
        if not (base_dir / 'base.blend').exists():
            shutil.copyfile(base, base_dir / 'base.blend.tmp')
            (base_dir / 'base.blend.tmp').replace(base_dir / 'base.blend')
        if (base_dir / 'base.blend').stat().st_size != base.stat().st_size:
            raise RuntimeError('Staged base scene size mismatch')
        kernel = Path('/root/.cache/cycles/kernels/cycles_kernel_sm_80_2AAF48C4D428C3FB1E8E567C96A28474.cubin')
        if not (kernel_dir / 'kernel_sm_80.cubin').exists():
            shutil.copyfile(kernel, kernel_dir / 'kernel_sm_80.cubin')
        shutil.copyfile(script, input_dir / 'candidate.py')
        runner_source = Path(__file__).with_name('blender_render.py')
        if not (runner_dir / 'render.py').exists():
            shutil.copyfile(runner_source, runner_dir / 'render.py')
        elif digest(runner_dir / 'render.py') != digest(runner_source):
            raise RuntimeError('Staged renderer changed; create a new run runtime')
        for filename in ['libcuda.so.1', 'libnvidia-ptxjitcompiler.so.1',
                         'libnvidia-nvvm.so.4', 'libnvidia-ml.so.1']:
            src = Path('/usr/lib/x86_64-linux-gnu') / filename
            if src.exists() and not (driver_dir / filename).exists():
                shutil.copyfile(src, driver_dir / filename)
    gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    views = task['views']
    if not views or any(not v.startswith('render') or not v.endswith('.png') or Path(v).name != v for v in views):
        raise ValueError('Invalid view name')
    cmd = ['docker', 'run', '--name', name, '--network', 'none', '--read-only',
           '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
           '--user', '65534:65534', '--pids-limit', '256', '--memory', '24g',
           '--cpus', '6', '--shm-size', '1g', '--tmpfs', '/tmp:rw,nosuid,size=2g',
           '--entrypoint', '/bin/bash', '--workdir', '/out',
           '-e', 'HOME=/tmp', '-e', 'OMP_NUM_THREADS=6', '-e', 'OPENBLAS_NUM_THREADS=1',
           '-e', 'MKL_NUM_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', 'CUDA_VISIBLE_DEVICES=' + gpu_uuid, '-e', 'LD_LIBRARY_PATH=/driver']
    physical_minor = None
    for information in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        text = information.read_text()
        if gpu_uuid in text:
            physical_minor = int(re.search(r'Device Minor:\s*(\d+)', text).group(1))
            break
    if physical_minor is None:
        raise RuntimeError('Cannot map CUDA index to Linux device minor')
    for device in [f'/dev/nvidia{physical_minor}', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']:
        if Path(device).exists():
            cmd += ['--device', device]
    mounts = [(blender.parent, '/opt/blender', 'ro'), (infinigen, '/opt/infinigen', 'ro'),
              (base_dir, '/task/base', 'ro'), (input_dir, '/task/input', 'ro'),
              (runner_dir, '/runtime', 'ro'),
              (kernel_dir, '/opt/blender/3.6/scripts/addons/cycles/lib', 'ro'), (driver_dir, '/driver', 'ro'),
              (output, '/out', 'rw')]
    for source, dest, mode in mounts:
        cmd += ['--mount', f'type=bind,src={source},dst={dest}' + (',readonly' if mode == 'ro' else '')]
    args = ['/opt/blender/blender', '--background', '--disable-autoexec', '--threads', '6',
            '/task/base/base.blend', '--python-exit-code', '3', '--python', '/runtime/render.py',
            '--', '--script', '/task/input/candidate.py', '--output', '/out',
            '--views', *views, '--samples', str(samples), '--seed', str(seed), '--infinigen', '/opt/infinigen']
    # Positional arguments avoid interpreting generated code as shell text.
    cmd += [IMAGE, '-c', 'exec "$@" > /out/blender.log 2>&1', 'blender-agent', *args]
    return cmd

@contextlib.contextmanager
def gpu_lock(lock_root, gpus):
    root = Path(lock_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        for gpu in gpus:
            file = (root / f'gpu-{gpu}.lock').open('a')
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                file.close()
                continue
            try:
                yield gpu
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
                file.close()
            return
        time.sleep(0.25)

def render(task, script, output, blender, infinigen, gpu, samples=512, seed=20261015, timeout=600):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o777)
    control = output.parent / 'sandbox-result.json'
    expected = {'script_sha256': digest(script),
                'runner_sha256': digest(Path(__file__).with_name('blender_render.py')),
                'sandbox_sha256': digest(__file__), 'base_blend': task['base_blend'],
                'views': task['views'], 'samples': samples, 'seed': seed, 'image': IMAGE,
                'kernel_sha256': digest('/root/.cache/cycles/kernels/cycles_kernel_sm_80_2AAF48C4D428C3FB1E8E567C96A28474.cubin')}
    if control.exists():
        old = json.loads(control.read_text())
        if all(old.get(k) == v for k, v in expected.items()):
            if not old['ok'] or all((output / v).is_file() for v in task['views']):
                return old
        raise RuntimeError('Sandbox cache provenance mismatch')
    name = 'bg-agent-' + uuid.uuid4().hex[:20]
    cmd = build_command(task, script, output, blender, infinigen, gpu, name, samples, seed)
    started = time.monotonic()
    timed_out = False
    docker_stdout = ''
    try:
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            rc, docker_stdout = process.returncode, (process.stdout + process.stderr)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(['docker', 'kill', name], capture_output=True, timeout=30)
            rc = -9
        inspect = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
        if inspect.returncode:
            raise RuntimeError('Docker infrastructure failure: ' + docker_stdout[-4000:] + inspect.stderr[-1000:])
        container = json.loads(inspect.stdout)[0]
        state = container['State']
        # The environment's Docker wrapper suppresses stdout, so collect the log through /out.
        if state.get('Error'):
            raise RuntimeError('Docker startup error: ' + state['Error'])
        if not timed_out:
            rc = state['ExitCode']
        log = (output / 'blender.log').read_text(errors='replace') if (output / 'blender.log').exists() else ''
        if rc == 0 and not (output / 'render.json').exists():
            rc = 3
        ok = rc == 0 and all((output / v).is_file() and not (output / v).is_symlink() for v in task['views'])
        if 'No CUDA device available' in log or 'error while loading shared libraries' in log:
            raise RuntimeError('Renderer infrastructure failure: ' + log[-3000:])
        result = {**expected, 'ok': ok, 'returncode': rc, 'timed_out': timed_out,
                  'oom_killed': state.get('OOMKilled', False), 'seconds': time.monotonic() - started,
                  'gpu': str(gpu), 'container_id': container['Id'], 'command': cmd,
                  'mounts': container['Mounts'], 'log_tail': log[-6000:]}
        if ok:
            from PIL import Image
            for view in task['views']:
                with Image.open(output / view) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise RuntimeError('Renderer output resolution mismatch')
            # Trusted host overwrites metadata; model code cannot choose score provenance.
            atomic_json(output / 'render.json', {**expected, 'ok': True, 'seconds': result['seconds']})
        else:
            (output / 'render.json').unlink(missing_ok=True)
        atomic_json(control, result)
        return result
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
