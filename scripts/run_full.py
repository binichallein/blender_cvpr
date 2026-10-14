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
        return (any(arg.endswith(('/infer_full.py', '/infer_replicas.py')) for arg in args)
                and '--variant' in args and args[args.index('--variant') + 1] == variant
                and '--output' in args and args[args.index('--output') + 1] == str(output))
    except (FileNotFoundError, IndexError, UnicodeError):
        return False


def replica_owned_gpus(pid, variant, output, allocated):
    """Release finished replica GPUs while retaining every live child's allocation."""
    path = Path(output) / 'replica-runs' / str(pid) / 'run.json'
    try:
        metadata = json.loads(path.read_text())
        children = metadata['children']
        declared = metadata['gpu_ids']
        # Missing, partial or inconsistent startup metadata must retain ownership.
        if (metadata['variant'] != variant or len(children) != len(declared)
                or len(set(declared)) != len(declared)
                or {child['gpu'] for child in children} != set(declared)
                or not set(allocated) <= set(declared)):
            return allocated
        live = {child['gpu'] for child in children
                if matching_process(child['pid'], variant, output)}
        return [gpu for gpu in allocated if gpu in live]
    except (OSError, ValueError, KeyError, TypeError):
        return allocated


class ExistingInference:
    def __init__(self, pid, variant, output):
        self.pid, self.variant, self.output = pid, variant, output

    def poll(self):
        # Coverage is checked separately; a lost parent's exit status is unavailable.
        return None if matching_process(self.pid, self.variant, self.output) else 0



def release_render_gpu(gpu, workers, releases):
    """Request release, but never claim a GPU while its render worker is alive."""
    if gpu not in workers:
        return True
    releases[gpu].set()
    if workers[gpu].is_alive():
        return False
    workers[gpu].join()
    del workers[gpu]
    del releases[gpu]
    return True


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
    parser.add_argument('--replicas', action='append', default=[], metavar='VARIANT=COUNT')
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
    replicas = {name: 1 for name in variants}
    for specification in a.replicas:
        name, value = specification.split('=', 1)
        count = int(value)
        assert name in variants and count >= 1
        assert count == 1 or variants[name]['tensor_parallel_size'] == 1
        replicas[name] = count
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
    workers, releases = {}, {}
    unseen = {(name, task['id']) for name in variants for task in tasks}

    def render_worker(gpu, release):
        while not stop.is_set() and not release.is_set():
            try:
                name, task = jobs.get(timeout=1)
            except queue.Empty:
                continue
            if stop.is_set() or release.is_set():
                jobs.put((name, task))
                jobs.task_done()
                break
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
                release = threading.Event()
                releases[gpu] = release
                worker = threading.Thread(target=render_worker, args=(gpu, release), daemon=True)
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
        driver = 'infer_replicas.py' if replicas[name] > 1 else 'infer_full.py'
        command = [a.inference_python, '-u', str(scripts / driver),
                   '--model', str(model), '--variant', name, '--manifest', a.manifest,
                   '--protocol', a.protocol, '--output', str(root / name / 'predictions')]
        if replicas[name] > 1:
            command.extend(['--replicas', str(replicas[name])])
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        log.close()
        running[name] = (child, gpus)
        previous_source = root / name / 'source.json'
        if previous_source.exists():
            previous = json.loads(previous_source.read_text())
            atomic_json(root / name / ('source-history-' + str(previous.get('inference_pid', 'unknown')) + '.json'),
                        previous)
        atomic_json(previous_source, {**source, 'gpu_ids': gpus, 'model_path': str(model),
                                      'inference_pid': child.pid, 'replicas': replicas[name]})
        print('INFERENCE_STARTED', name, child.pid, ','.join(gpus), flush=True)

    last_status = 0
    references_ready = False
    try:
        while True:
            discover()
            for name, (child, gpus) in list(running.items()):
                status = child.poll()
                if status is None:
                    active_gpus = replica_owned_gpus(child.pid, name, root / name / 'predictions', gpus)
                    if active_gpus != gpus:
                        print('REPLICA_GPUS_RELEASED', name, sorted(set(gpus)-set(active_gpus)), flush=True)
                        running[name] = (child, active_gpus)
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
            reference_active = process_alive(a.reference_pid)
            if not reference_active and not references_ready:
                references_ready = all(
                    (Path(a.references) / task['id'] / kind / 'render.json').exists()
                    for task in tasks for kind in ['start', 'goal'])
            inference_pool = ['2', '3', '0', '1'] if references_ready else ['2', '3']
            render_pool = ['2', '3'] if reference_active else ['0', '1', '2', '3']
            used = {gpu for _, gpus in running.values() for gpu in gpus}
            free = [gpu for gpu in inference_pool if gpu not in used]
            reserved = set()
            for name in list(pending):
                variant = variants[name]
                count = variant['tensor_parallel_size'] * replicas[name]
                if len(free) < count:
                    continue
                model = model_path(a.weights_root, variant['model'])
                if model is None:
                    continue
                chosen, free = free[:count], free[count:]
                reserved.update(chosen)
                released = [release_render_gpu(gpu, workers, releases) for gpu in chosen]
                if not all(released):
                    continue
                launch(name, model, chosen)
                pending.remove(name)
            # Download waits should not leave inference GPUs idle. Render one task
            # at a time there; a ready model reserves them and waits for workers to exit.
            used = {gpu for _, gpus in running.values() for gpu in gpus}
            if not jobs.empty():
                start_render_workers([gpu for gpu in render_pool if gpu not in used and gpu not in reserved])
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
                'render_gpu_ids': sorted(workers),
                'render_gpu_releasing': sorted(gpu for gpu, event in releases.items() if event.is_set()),
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
