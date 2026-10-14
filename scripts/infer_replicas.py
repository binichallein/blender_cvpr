"""Distribute unfinished tasks across identical single-GPU inference replicas."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def partition_pending(tasks, completed, replicas):
    if replicas < 1:
        raise ValueError('replicas must be positive')
    pending = [task for task in tasks if task not in completed]
    return [pending[index::replicas] for index in range(replicas)]


def main():
    parser = argparse.ArgumentParser()
    for key in ['model', 'variant', 'manifest', 'protocol', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--replicas', type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    variant = next(v for v in protocol['variants'] if v['id'] == args.variant)
    if variant['tensor_parallel_size'] != 1:
        raise ValueError('Replicas currently require single-GPU model engines')
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(gpu_ids) != args.replicas or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Each replica requires an exclusively allocated GPU')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    tasks = [t['id'] for t in json.loads(Path(args.manifest).read_text())['all_tasks']]
    completed = {task for task in tasks if (root / task / 'generation.json').exists()}
    groups = partition_pending(tasks, completed, args.replicas)
    location = root / 'replica-runs' / str(os.getpid())
    location.mkdir(parents=True)
    metadata = {'variant': args.variant, 'replicas': args.replicas, 'gpu_ids': gpu_ids,
                'protocol_sha256': hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
                'started_unix': time.time(), 'previously_completed_tasks': sorted(completed),
                'task_groups': groups, 'children': []}
    record = location / 'run.json'
    children = []
    def save():
        record.write_text(json.dumps(metadata, indent=2))
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, (gpu, group) in enumerate(zip(gpu_ids, groups)):
            if not group:
                continue
            task_file = location / (str(index) + '-tasks.json')
            task_file.write_text(json.dumps(group))
            command = [sys.executable, '-u', str(Path(__file__).with_name('infer_full.py'))]
            for key in ['model', 'variant', 'manifest', 'protocol', 'output']:
                command.extend(['--' + key, getattr(args, key)])
            command.extend(['--task-ids-file', str(task_file)])
            log_path = location / (str(index) + '.log')
            with log_path.open('a') as log:
                child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                         stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            metadata['children'].append({'pid': child.pid, 'gpu': gpu, 'log': str(log_path),
                                         'assigned_tasks': len(group)})
            print('REPLICA_STARTED', args.variant, index, child.pid, gpu, len(group), flush=True)
        save()
        while any(child.poll() is None for child in children):
            failures = [(child.pid, child.returncode) for child in children
                        if child.poll() not in (None, 0)]
            if failures:
                raise RuntimeError('Replica infrastructure failure: ' + repr(failures))
            time.sleep(2)
        if any(child.returncode for child in children):
            raise RuntimeError('A replica failed')
        missing = [task for task in tasks if not (root / task / 'generation.json').exists()]
        if missing:
            raise RuntimeError('Replica coverage incomplete: ' + repr(missing))
        metadata.update(finished_unix=time.time(), complete=True)
        save()
        print('REPLICAS_COMPLETE', args.variant, flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        metadata['returncodes'] = {str(child.pid): child.returncode for child in children}
        save()


if __name__ == '__main__':
    main()
