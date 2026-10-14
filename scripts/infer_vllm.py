"""Continuous one-shot inference; prompts and frozen scorer match infer.py."""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
import torch
import transformers
import vllm
from PIL import Image
from transformers import AutoProcessor
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from async_queue import consume
from infer import build_messages, extract_code, SYSTEM, CATEGORY


def prepare(task, processor, dest):
    messages = build_messages(task)
    (dest / 'input.json').write_text(json.dumps(messages, indent=2))
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = []
    try:
        for message in messages:
            for item in message['content']:
                if item['type'] == 'image':
                    with Image.open(item['url']) as source:
                        images.append(source.convert('RGB'))
        return {'prompt': prompt, 'multi_modal_data': {'image': images}}, images
    except BaseException:
        for image in images:
            image.close()
        raise


async def run(a):
    torch.set_num_threads(1)
    protocol_path = Path(a.protocol)
    protocol = json.loads(protocol_path.read_text())
    assert protocol['frozen']
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text())
    ids = manifest['subset_ids'] if a.split == 'subset' else [t['id'] for t in manifest['all_tasks']]
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {
        'model': Path(a.model).name,
        'model_config_sha256': hashlib.sha256((Path(a.model) / 'config.json').read_bytes()).hexdigest(),
        'protocol_sha256': protocol_sha,
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'backend': 'vllm.AsyncLLM', 'vllm': vllm.__version__,
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'dtype': 'bfloat16', 'temperature': 0, 'max_new_tokens': protocol['max_new_tokens'],
        'seed': 20261015, 'max_model_len': 16384, 'max_num_seqs': 4,
        'max_in_flight_requests': 8, 'scheduling': 'continuous, replenish on each completion',
        'enforce_eager': True, 'gpu_memory_utilization': 0.9,
        'max_num_batched_tokens': 8192, 'max_images': 10, 'max_image_pixels': 1048576,
        'system_prompt': SYSTEM, 'category_instructions': CATEGORY,
        'input_views': 'all matched current/goal views with valid base-scene cameras',
        'repairs': 0, 'attempts': 1,
    }
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Refusing mismatched resume'
    else:
        config_path.write_text(json.dumps(config, indent=2))
    pending = [t for t in manifest['all_tasks'] if t['id'] in ids
               and not (out / t['id'] / 'generation.json').exists()]
    if not pending:
        return
    print('LOADING', Path(a.model).name, len(pending), flush=True)
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    args = AsyncEngineArgs(
        model=a.model, dtype='bfloat16', max_model_len=16384, max_num_seqs=4,
        gpu_memory_utilization=0.9, enforce_eager=True, max_num_batched_tokens=8192,
        limit_mm_per_prompt={'image': 10}, mm_processor_kwargs={'max_pixels': 1048576},
        seed=20261015, enable_log_requests=False,
    )
    engine = AsyncLLM.from_engine_args(args)
    started = time.monotonic()

    async def generate(task):
        dest = out / task['id']
        dest.mkdir(exist_ok=True)
        request, images = await asyncio.to_thread(prepare, task, processor, dest)
        request_start = time.monotonic()
        print('SUBMITTED', task['id'], flush=True)
        try:
            # Every task gets one request, with no retry or corrective feedback.
            params = SamplingParams(temperature=0, max_tokens=protocol['max_new_tokens'],
                                    seed=20261015, output_kind=RequestOutputKind.FINAL_ONLY)
            result = None
            async for output in engine.generate(
                    prompt=request, sampling_params=params, request_id=task['id']):
                result = output
            if result is None or not result.finished or len(result.outputs) != 1:
                raise RuntimeError('Incomplete inference output: ' + task['id'])
            answer = result.outputs[0]
            if answer.finish_reason not in ('stop', 'length'):
                raise RuntimeError('Infrastructure terminated request: ' + str(answer.finish_reason))
            raw = answer.text
            (dest / 'response.txt').write_text(raw)
            row = {
                'task': task['id'], 'category': task['category'], 'ok': False,
                'protocol_sha256': protocol_sha,
                'input_tokens': len(result.prompt_token_ids), 'output_tokens': len(answer.token_ids),
                'truncated': answer.finish_reason == 'length', 'finish_reason': answer.finish_reason,
                'seconds': time.monotonic() - request_start,
                'submitted_at_elapsed_seconds': request_start - started,
                'completed_at_elapsed_seconds': time.monotonic() - started,
            }
            try:
                if row['truncated']:
                    raise ValueError('Generation reached token budget')
                (dest / 'prediction.py').write_text(extract_code(raw))
                row['ok'] = True
            except (ValueError, SyntaxError) as error:
                row['error'] = repr(error)
            temp = dest / 'generation.json.tmp'
            temp.write_text(json.dumps(row, indent=2))
            temp.replace(dest / 'generation.json')
            print(json.dumps(row), flush=True)
        finally:
            for image in images:
                image.close()

    print('MODEL_READY', flush=True)
    try:
        await consume(pending, generate, workers=8)
        (out / (a.split + '-timing.json')).write_text(json.dumps({
            'completed_in_this_invocation': len(pending),
            'generation_wall_seconds': time.monotonic() - started,
        }, indent=2))
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    for key in ['model', 'manifest', 'protocol', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--split', choices=['subset', 'all'], default='subset')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
