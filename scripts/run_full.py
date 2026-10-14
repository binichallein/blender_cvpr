"""Resume all six full-set variants; overlap inference, rendering and scoring."""
import argparse
import fcntl
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from render_one import render


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def process_alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False



def matching_process(pid, variant, output, proc_root='/proc'):
    try:
        process = Path(proc_root) / str(pid)
        if process.joinpath('stat').read_text().split()[2] == 'Z':
            return False
        args = process.joinpath('cmdline').read_bytes().decode().split('\0')
        return (any(arg.endswith('/infer_full.py') for arg in args)
                and '--variant' in args and args[args.index('--variant') + 1] == variant
                and '--output' in args and args[args.index('--output') + 1] == str(output))
    except (FileNotFoundError, IndexError, UnicodeError):
        return False


class ExistingInference:
    def __init__(self, pid, variant, output):
        self.pid, self.variant, self.output = pid, variant, output

    def poll(self):
        # Coverage is checked separately; a lost parent's exit status is unavailable.
        return None if matching_process(self.pid, self.variant, self.output) else 0


def model_path(root, model):
    name = model.split('/')[-1]
    for candidate in [Path(root) / name, Path(root) / name.replace('.', '___')]:
        required = ['config.json', 'tokenizer_config.json', 'preprocessor_config.json',
                    'model.safetensors.index.json']
        if not all((candidate / f).exists() for f in required):
            continue
        if not ((candidate / 'tokenizer.json').exists()
                or all((candidate / f).exists() for f in ['vocab.json', 'merges.txt'])):
            continue
        try:
            index = json.loads((candidate / 'model.safetensors.index.json').read_text())
            shards = set(index['weight_map'].values())
            if shards and all((candidate / shard).is_file() and (candidate / shard).stat().st_size > 0
                              for shard in shards):
                return candidate
        except (OSError, ValueError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    for key in ['manifest', 'protocol', 'output', 'blender', 'infinigen', 'clip',
                'references', 'weights-root', 'inference-python']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--reference-pid', type=int, required=True)
    parser.add_argument('--ipc-directory', default='/dev/shm/bg-full')
    a = parser.parse_args()
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    Path(a.ipc_directory).mkdir(parents=True, exist_ok=True)
    lockfile = (root / 'coordinator.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = Path(__file__).resolve().parent
    manifest = json.loads(Path(a.manifest).read_text())
    protocol_path = Path(a.protocol)
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert protocol['frozen'] and protocol['scope'].startswith('all released')
    tasks = manifest['all_tasks']
    task_by_id = {task['id']: task for task in tasks}
    variants = {v['id']: v for v in protocol['variants']}
    source = {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=scripts.parent, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=scripts.parent, text=True),
        'protocol_sha256': protocol_sha, 'started_unix': time.time(),
    }
    source_path = root / ('source-' + str(os.getpid()) + '.json')
    atomic_json(source_path, source)
    pending = ['qwen3-vl-32b-nothink', 'qwen3-vl-8b-nothink', 'qwen3-vl-8b-think',
               'qwen3.6-27b-nothink', 'qwen3.6-27b-think', 'qwen3-vl-32b-think']
    for name in variants:
        (root / name / 'predictions').mkdir(parents=True, exist_ok=True)
    running, finished, inference_failed = {}, set(), {}
    for name in variants:
        previous_source = root / name / 'source.json'
        if not previous_source.exists():
            continue
        previous = json.loads(previous_source.read_text())
        pid = previous.get('inference_pid')
        output = root / name / 'predictions'
        if pid and matching_process(pid, name, output):
            assert previous['protocol_sha256'] == protocol_sha
            gpus = previous['gpu_ids']
            assert not (set(gpus) & {gpu for _, other in running.values() for gpu in other})
            running[name] = (ExistingInference(pid, name, output), gpus)
            pending.remove(name)
            print('INFERENCE_ADOPTED', name, pid, flush=True)
    score_jobs, scored, scoring_failed = {}, set(), {}
    for name in variants:
        result_path = root / name / 'all-scores.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result['protocol_sha256'] == protocol_sha
            assert {r['task'] for r in result['rows']} == set(task_by_id)
            scored.add(name)
    jobs = queue.Queue()
    render_done = {name: set() for name in variants}
    render_errors = []
    state_lock = threading.Lock()
    stop = threading.Event()
    workers = {}
    unseen = {(name, task['id']) for name in variants for task in tasks}

    def render_worker(gpu):
        while not stop.is_set():
            try:
                name, task = jobs.get(timeout=1)
            except queue.Empty:
                continue
            try:
                destination = root / name / 'renders' / task['id'] / 'prediction'
                result = render(task, root / name / 'predictions' / task['id'] / 'prediction.py',
                                destination, a.blender, a.infinigen, gpu, reuse_failures=True)
                print('RENDER', json.dumps({'variant': name, 'task': task['id'],
                                            'ok': result['ok']}), flush=True)
                with state_lock:
                    render_done[name].add(task['id'])
            except Exception as error:
                with state_lock:
                    render_errors.append({'variant': name, 'task': task['id'], 'error': repr(error)})
            finally:
                jobs.task_done()

    def start_render_workers(gpus):
        for gpu in gpus:
            if gpu not in workers:
                worker = threading.Thread(target=render_worker, args=(gpu,), daemon=True)
                workers[gpu] = worker
                worker.start()
                print('RENDER_WORKER_STARTED', gpu, flush=True)

    def discover():
        for name, task_id in sorted(list(unseen)):
            path = root / name / 'predictions' / task_id / 'generation.json'
            if not path.exists():
                continue
            generation = json.loads(path.read_text())
            assert generation['protocol_sha256'] == protocol_sha
            assert generation['task'] == task_id
            if generation['ok'] and name not in scored:
                jobs.put((name, task_by_id[task_id]))
            else:
                with state_lock:
                    render_done[name].add(task_id)
            unseen.remove((name, task_id))

    def inference_complete(name):
        return not any(variant == name for variant, _ in unseen)

    def launch(name, model, gpus):
        variant = variants[name]
        log = (root / 'logs' / (name + '-inference.log')).open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
                   VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1',
                   TMPDIR=a.ipc_directory)
        command = [a.inference_python, '-u', str(scripts / 'infer_full.py'),
                   '--model', str(model), '--variant', name, '--manifest', a.manifest,
                   '--protocol', a.protocol, '--output', str(root / name / 'predictions')]
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        log.close()
        running[name] = (child, gpus)
        atomic_json(root / name / 'source.json', {**source, 'gpu_ids': gpus, 'model_path': str(model),
                                                'inference_pid': child.pid})
        print('INFERENCE_STARTED', name, child.pid, ','.join(gpus), flush=True)

    last_status = 0
    references_ready = False
    try:
        while True:
            discover()
            for name, (child, gpus) in list(running.items()):
                status = child.poll()
                if status is None:
                    continue
                del running[name]
                discover()
                if status == 0 and inference_complete(name):
                    finished.add(name)
                    print('INFERENCE_DONE', name, flush=True)
                else:
                    inference_failed[name] = {'returncode': status,
                        'remaining': sum(v == name for v, _ in unseen)}
                    print('INFERENCE_FAILED', name, inference_failed[name], flush=True)
            for name in list(pending):
                if inference_complete(name):
                    finished.add(name)
                    pending.remove(name)
            used = {gpu for _, gpus in running.values() for gpu in gpus}
            free = [gpu for gpu in ['2', '3'] if gpu not in used]
            for name in list(pending):
                variant = variants[name]
                count = variant['tensor_parallel_size']
                if len(free) < count:
                    continue
                model = model_path(a.weights_root, variant['model'])
                if model is None:
                    continue
                chosen, free = free[:count], free[count:]
                launch(name, model, chosen)
                pending.remove(name)
            if not process_alive(a.reference_pid):
                start_render_workers(['0', '1'])
                references_ready = all(
                    (Path(a.references) / task['id'] / kind / 'render.json').exists()
                    for task in tasks for kind in ['start', 'goal'])
            if not pending and not running:
                # Inference has released both GPUs; use all four for remaining renders.
                start_render_workers(['2', '3'])
            with state_lock:
                render_counts = {name: len(done) for name, done in render_done.items()}
                errors = list(render_errors)
            for name in finished:
                if (name in scored or name in score_jobs or name in scoring_failed
                        or render_counts[name] != len(tasks) or not references_ready):
                    continue
                command = [sys.executable, '-u', str(scripts / 'score.py')]
                options = {'manifest': a.manifest, 'protocol': a.protocol,
                           'references': a.references, 'predictions': str(root / name / 'predictions'),
                           'renders': str(root / name / 'renders'), 'clip': a.clip,
                           'output': str(root / name / 'all-scores.json'), 'split': 'all'}
                for key, value in options.items():
                    command.extend(['--' + key, value])
                with (root / 'logs' / (name + '-score.log')).open('a') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                score_jobs[name] = child
                print('SCORING_STARTED', name, child.pid, flush=True)
            for name, child in list(score_jobs.items()):
                status = child.poll()
                if status is None:
                    continue
                del score_jobs[name]
                if status == 0:
                    scored.add(name)
                    print('SCORE_DONE', name, flush=True)
                else:
                    scoring_failed[name] = status
            if len(scored) == len(variants):
                result = {name: json.loads((root / name / 'all-scores.json').read_text())
                          for name in variants}
                atomic_json(root / 'complete.json', result)
                print('FULL_MATRIX_COMPLETE', flush=True)
                break
            status = {
                'unix_time': time.time(), 'references_ready': references_ready,
                'pending_weights_or_gpu': pending,
                'inference_running': {name: {'pid': child.pid, 'gpus': gpus}
                                      for name, (child, gpus) in running.items()},
                'generated': {name: len(tasks) - sum(v == name for v, _ in unseen) for name in variants},
                'render_handled': render_counts, 'render_queue': jobs.qsize(),
                'inference_failed': inference_failed, 'render_infrastructure_errors': errors,
                'scoring_running': list(score_jobs), 'scored': sorted(scored),
                'scoring_failed': scoring_failed,
            }
            atomic_json(root / 'status.json', status)
            if time.monotonic() - last_status > 45:
                print('STATUS', json.dumps(status), flush=True)
                last_status = time.monotonic()
            if not pending and not running and not score_jobs and jobs.empty():
                if inference_failed or scoring_failed or errors:
                    raise RuntimeError('Unfinished matrix; inspect status.json and resume after fixing infrastructure')
                if not references_ready and not process_alive(a.reference_pid):
                    raise RuntimeError('Full reference renders incomplete; inspect reference logs')
            time.sleep(3)
    finally:
        stop.set()
        for worker in workers.values():
            worker.join(timeout=610)
        # Do not signal inference workers on a Python exception: their atomic outputs
        # remain useful and are adopted after process-identity validation on resume.


if __name__ == '__main__':
    main()
