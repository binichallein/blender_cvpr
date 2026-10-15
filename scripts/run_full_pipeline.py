"""Verify downloaded checkpoints, run a frozen matrix, audit it, and export a report."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_full import atomic_json, model_path


def verify_checkpoint(directory, inventory):
    files = inventory['files']
    checked = []
    for item in files:
        if item.get('Type') != 'blob':
            continue
        relative = Path(item['Path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint inventory path')
        path = directory / relative
        if not path.is_file() or path.stat().st_size != item['Size']:
            raise ValueError('Missing or wrong-size checkpoint file: ' + str(relative))
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != item['Sha256']:
            raise ValueError('Checkpoint SHA256 mismatch: ' + str(relative))
        checked.append({'path': str(relative), 'size': item['Size'], 'sha256': digest})
        print('VERIFIED', relative, flush=True)
    assert checked, 'Empty checkpoint inventory'
    return {'passed': True, 'model': inventory['model'], 'files': checked}


def download_running(pid_file, model):
    pid = int(Path(pid_file).read_text())
    process = Path('/proc') / str(pid)
    try:
        args = process.joinpath('cmdline').read_bytes().decode().split('\0')
        return (process.joinpath('stat').read_text().split()[2] != 'Z'
                and any(arg.endswith('/download_weights.py') for arg in args)
                and model in args)
    except FileNotFoundError:
        return False


def run(config):
    root = Path(config['run'])
    repo = Path(config['repo'])
    scripts = repo / 'scripts'
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    models = list(dict.fromkeys(v['model'] for v in protocol['variants']))
    assert len(models) == 1, 'This pipeline expects one checkpoint with multiple native modes'
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    status_path = root / 'pipeline-status.json'

    def phase(name, **extra):
        atomic_json(status_path, {'phase': name, 'unix_time': time.time(),
                                'protocol_sha256': protocol_sha, **extra})
        print('PHASE', name, flush=True)

    def command(script, options):
        cmd = [sys.executable, '-u', str(scripts / script)]
        for key, value in options.items():
            cmd.extend(['--' + key, str(value)])
        subprocess.run(cmd, cwd=repo, check=True)

    phase('waiting_for_download')
    while download_running(root / 'download.pid', models[0]):
        time.sleep(15)
    directory = model_path(config['weights_root'], models[0])
    if directory is None:
        raise RuntimeError('Download stopped without a complete checkpoint; inspect download.log')
    phase('verifying_checkpoint')
    inventory = json.loads(Path(config['model_inventory']).read_text())
    assert inventory['model'] == models[0]
    verified = verify_checkpoint(directory, inventory)
    verified['protocol_sha256'] = protocol_sha
    atomic_json(root / 'checkpoint-audit.json', verified)
    phase('evaluating')
    command('run_full.py', {
        'manifest': config['manifest'], 'protocol': protocol_path,
        'output': root / 'matrix', 'blender': config['blender'],
        'infinigen': config['infinigen'], 'clip': config['clip'],
        'references': config['references'], 'weights-root': config['weights_root'],
        'inference-python': config['inference_python'], 'reference-pid': 0,
        'ipc-directory': config['ipc_directory'],
    })
    phase('auditing_results')
    command('audit_full.py', {
        'runs': root / 'matrix', 'manifest': config['manifest'], 'protocol': protocol_path,
        'references': config['references'], 'output': root / 'final-audit.json',
    })
    phase('exporting_report')
    output = Path(config['report_output'])
    output.mkdir(parents=True, exist_ok=True)
    for filename in ['final-audit.json', 'checkpoint-audit.json']:
        shutil.copy2(root / filename, output / filename)
    command('report_full.py', {
        'runs': root / 'matrix', 'manifest': config['manifest'], 'protocol': protocol_path,
        'output': output, 'model-metadata': root / 'model-metadata',
        'reference-audit': config['reference_audit'],
    })
    atomic_json(root / 'postprocess-complete.json', {
        'passed': True, 'finished_unix': time.time(), 'output': str(output),
        'protocol_sha256': protocol_sha,
    })
    phase('complete', output=str(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        run(config)
    except BaseException as error:
        atomic_json(Path(config['run']) / 'pipeline-error.json',
                    {'error': repr(error), 'unix_time': time.time()})
        raise


if __name__ == '__main__':
    main()
