"""Host-side renderer wrapper; all paths are supplied by arguments."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

def render(task, script, output, blender, infinigen, gpu, samples=512, seed=20261015, timeout=600, views=None):
    output = Path(output).resolve()
    script = Path(script).resolve()
    blender = Path(blender).resolve()
    infinigen = Path(infinigen).resolve()
    output.mkdir(parents=True, exist_ok=True)
    views = views or task['views']
    meta = output/'render.json'
    expected = {'views': views, 'samples': samples, 'seed': seed,
                'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).with_name('blender_render.py').read_bytes()).hexdigest(),
                'base_blend': task['base_blend']}
    if meta.exists():
        previous = json.loads(meta.read_text())
        if previous.get('ok') and all(previous.get(k)==v for k,v in expected.items()) and all((output/v).exists() for v in views):
            return previous
    meta.unlink(missing_ok=True)
    for view in views:
        (output/view).unlink(missing_ok=True)
    runner = Path(__file__).with_name('blender_render.py')
    command = [str(blender), '--background', '--disable-autoexec', '--threads', '6',
               task['base_blend'], '--python-exit-code', '3', '--python', str(runner),
               '--', '--script', str(script), '--output', str(output), '--views', *views,
               '--samples', str(samples), '--seed', str(seed), '--infinigen', str(infinigen)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='6',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    with (output/'blender.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True, cwd=str(output))
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = -9
    if rc != 0 or not meta.exists() or not all((output/v).exists() for v in views):
        result = {'ok': False, 'returncode': rc, 'seconds': time.monotonic()-start,
                  **expected, 'command': command, 'script': str(script)}
        meta.unlink(missing_ok=True)
        (output/'failure.json').write_text(json.dumps(result, indent=2))
        return result
    result = {**json.loads(meta.read_text()), **expected}
    meta.write_text(json.dumps(result, indent=2))
    (output/'failure.json').unlink(missing_ok=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--kind', default='goal')
    ap.add_argument('--script')
    ap.add_argument('--output', required=True)
    ap.add_argument('--blender', required=True)
    ap.add_argument('--infinigen', required=True)
    ap.add_argument('--gpu', default='3')
    ap.add_argument('--samples',type=int,default=512)
    ap.add_argument('--seed',type=int,default=20261015)
    ap.add_argument('--views',nargs='*')
    args=ap.parse_args()
    tasks=json.loads(Path(args.manifest).read_text())['all_tasks']
    task=next(t for t in tasks if t['id']==args.task)
    script=args.script or str(Path(task['directory'])/(args.kind+'.py'))
    result=render(task,script,args.output,args.blender,args.infinigen,args.gpu,
                  args.samples,args.seed,views=args.views)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['ok'] else 1)

if __name__=='__main__':
    main()
