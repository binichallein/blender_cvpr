"""Full-set native thinking comparison with continuously replenished vLLM requests."""
import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
import transformers
import vllm
from PIL import Image
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from async_queue import consume
from infer import build_messages, extract_code, SYSTEM, CATEGORY
from thinking_output import split_output


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def template_text(path):
    path = Path(path)
    if (path / 'chat_template.jinja').exists():
        return (path / 'chat_template.jinja').read_text()
    if (path / 'chat_template.json').exists():
        return json.loads((path / 'chat_template.json').read_text())['chat_template']
    value = json.loads((path / 'tokenizer_config.json').read_text()).get('chat_template')
    if not isinstance(value, str):
        raise ValueError('Cannot locate the official chat template')
    return value


def format_prompt(messages, tokenizer, template, variant):
    kwargs = {'enable_thinking': variant['thinking']} if variant['family'] in ('qwen3.5', 'qwen3.6') else {}
    prompt = tokenizer.apply_chat_template(messages, chat_template=template,
        tokenize=False, add_generation_prompt=True, **kwargs)
    is_open = prompt.rstrip().endswith('<think>')
    if is_open != variant['thinking']:
        raise ValueError('Native thinking prefix does not match requested mode')
    return prompt


def prepare(task, tokenizer, template, variant, destination):
    messages = build_messages(task)
    atomic_json(destination / 'input.json', messages)
    prompt = format_prompt(messages, tokenizer, template, variant)
    images = []
    try:
        for message in messages:
            for item in message['content']:
                if item['type'] == 'image':
                    with Image.open(item['url']) as image:
                        images.append(image.convert('RGB'))
        return {'prompt': prompt, 'multi_modal_data': {'image': images}}, images
    except BaseException:
        for image in images:
            image.close()
        raise


def persist_generation(completion, task, destination, tokenizer, variant, protocol_sha):
    raw = completion['text']
    (destination / 'response.txt').write_text(raw)
    row = {
        'task': task['id'], 'category': task['category'], 'ok': False,
        'protocol_sha256': protocol_sha, 'thinking': variant['thinking'],
        'input_tokens': completion['input_tokens'],
        'output_tokens': len(completion['token_ids']),
        'finish_reason': completion['finish_reason'],
        'truncated': completion['finish_reason'] == 'length',
        'seconds': completion['seconds'],
        'submitted_at_elapsed_seconds': completion['submitted_at_elapsed_seconds'],
        'completed_at_elapsed_seconds': completion['completed_at_elapsed_seconds'],
    }
    try:
        parts = split_output(completion['token_ids'], tokenizer, variant['thinking'])
        row.update(reasoning_tokens=parts['reasoning_tokens'], answer_tokens=parts['answer_tokens'],
                   reasoning_terminated=True)
        (destination / 'reasoning.txt').write_text(parts['reasoning'])
        (destination / 'final.txt').write_text(parts['final'])
        if row['truncated']:
            raise ValueError('Generation reached total token budget')
        (destination / 'prediction.py').write_text(extract_code(parts['final']))
        row['ok'] = True
    except (ValueError, SyntaxError) as error:
        row['error'] = str(error)
        row.setdefault('reasoning_terminated', False)
        if variant['thinking'] and tokenizer.convert_tokens_to_ids('</think>') not in completion['token_ids']:
            row.update(reasoning_tokens=len(completion['token_ids']), answer_tokens=0)
            (destination / 'reasoning.txt').write_text(raw)
    atomic_json(destination / 'generation.json', row)
    print(json.dumps(row), flush=True)


async def run(a):
    torch.set_num_threads(1)
    protocol_path = Path(a.protocol)
    protocol = json.loads(protocol_path.read_text())
    assert protocol['frozen'] and protocol['max_new_tokens'] == 32768
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    variant = next(v for v in protocol['variants'] if v['id'] == a.variant)
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text())
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    template = template_text(a.model)
    # Validate native mode before any benchmark query is submitted.
    format_prompt([{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}],
                  tokenizer, template, variant)
    config = {
        'variant': variant, 'model': variant['model'],
        'model_config_sha256': hashlib.sha256((Path(a.model) / 'config.json').read_bytes()).hexdigest(),
        'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
        'protocol_sha256': protocol_sha,
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'backend': 'vllm.AsyncLLM', 'vllm': vllm.__version__,
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'dtype': 'bfloat16', 'sampling': protocol['sampling'],
        'max_new_tokens': protocol['max_new_tokens'], 'max_model_len': protocol['context_capacity'],
        'max_num_seqs': 16, 'max_in_flight_requests': 32,
        'enforce_eager': True, 'gpu_memory_utilization': 0.9,
        'max_num_batched_tokens': 8192, 'max_images': 10, 'max_image_pixels': 1048576,
        'system_prompt': SYSTEM, 'category_instructions': CATEGORY,
        'repairs': 0, 'attempts': 1, 'scheduling': 'continuous; replenish on each completion',
    }
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Refusing mismatched resume'
    else:
        atomic_json(config_path, config)
    requested_ids = {task['id'] for task in manifest['all_tasks']}
    if a.task_ids_file:
        selected = json.loads(Path(a.task_ids_file).read_text())
        assert len(selected) == len(set(selected)) and set(selected) <= requested_ids
        requested_ids = set(selected)
    atomic_json(out / ('invocation-' + str(os.getpid()) + '.json'), {
        'started_unix': time.time(), 'gpu_ids': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'task_ids': sorted(requested_ids), 'protocol_sha256': protocol_sha,
        'task_ids_file': a.task_ids_file,
    })
    pending = []
    for task in manifest['all_tasks']:
        if task['id'] not in requested_ids:
            continue
        destination = out / task['id']
        if (destination / 'generation.json').exists():
            continue
        if (destination / 'completion.json').exists():
            persist_generation(json.loads((destination / 'completion.json').read_text()),
                               task, destination, tokenizer, variant, protocol_sha)
        else:
            pending.append(task)
    if not pending and not a.smoke_only:
        return
    print('LOADING', variant['id'], 'pending', len(pending), flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=a.model, dtype='bfloat16', max_model_len=protocol['context_capacity'],
        max_num_seqs=16, gpu_memory_utilization=0.9, enforce_eager=True,
        max_num_batched_tokens=8192, tensor_parallel_size=variant['tensor_parallel_size'],
        limit_mm_per_prompt={'image': 10}, mm_processor_kwargs={'max_pixels': 1048576},
        seed=protocol['sampling']['seed'], enable_log_requests=False,
    ))
    started = time.monotonic()

    async def request(prompt, request_id, limit):
        sampling = SamplingParams(**protocol['sampling'], max_tokens=limit,
                                  output_kind=RequestOutputKind.FINAL_ONLY)
        result = None
        async for output in engine.generate(prompt=prompt, sampling_params=sampling, request_id=request_id):
            result = output
        if result is None or not result.finished or len(result.outputs) != 1:
            raise RuntimeError('Incomplete engine output: ' + request_id)
        answer = result.outputs[0]
        if answer.finish_reason not in ('stop', 'length'):
            raise RuntimeError('Infrastructure terminated request: ' + str(answer.finish_reason))
        return result, answer

    async def smoke():
        colors = [('red', (255, 0, 0)), ('blue', (0, 0, 255)),
                  ('green', (0, 255, 0)), ('yellow', (255, 255, 0))]
        async def one(index):
            name, rgb = colors[index % len(colors)]
            messages = [{'role': 'user', 'content': [{'type': 'image'},
                {'type': 'text', 'text': 'Name the solid color in this image. Give the color in your final answer.'}]}]
            prompt = format_prompt(messages, tokenizer, template, variant)
            image = Image.new('RGB', (128, 128), rgb)
            try:
                _, answer = await request({'prompt': prompt, 'multi_modal_data': {'image': [image]}},
                                          'smoke-' + str(index), 2048)
                parts = split_output(answer.token_ids, tokenizer, variant['thinking'])
                assert answer.finish_reason == 'stop'
                assert re.search(r'\b' + name + r'\b', parts['final'].lower()), parts['final']
                return {'color': name, 'final': parts['final'],
                        'reasoning_tokens': parts['reasoning_tokens'], 'output_tokens': len(answer.token_ids)}
            finally:
                image.close()
        rows = await asyncio.gather(*(one(i) for i in range(8)))
        smoke_result = {'variant': variant['id'], 'passed': True, 'results': rows}
        atomic_json(out / ('smoke-' + str(os.getpid()) + '.json'), smoke_result)
        atomic_json(out / 'smoke.json', smoke_result)
        print('SMOKE_PASSED', variant['id'], flush=True)

    async def generate(task):
        destination = out / task['id']
        destination.mkdir(exist_ok=True)
        prompt, images = await asyncio.to_thread(prepare, task, tokenizer, template, variant, destination)
        request_start = time.monotonic()
        print('SUBMITTED', task['id'], flush=True)
        try:
            result, answer = await request(prompt, task['id'], protocol['max_new_tokens'])
            completion = {
                'text': answer.text, 'token_ids': list(answer.token_ids),
                'input_tokens': len(result.prompt_token_ids), 'finish_reason': answer.finish_reason,
                'seconds': time.monotonic() - request_start,
                'submitted_at_elapsed_seconds': request_start - started,
                'completed_at_elapsed_seconds': time.monotonic() - started,
            }
            # Save the full engine output before parsing, so a parser/process interruption
            # can be recovered without asking the model for another answer.
            atomic_json(destination / 'completion.json', completion)
            persist_generation(completion, task, destination, tokenizer, variant, protocol_sha)
        finally:
            for image in images:
                image.close()

    try:
        await smoke()
        if a.smoke_only:
            return
        started = time.monotonic()
        generation_started_unix = time.time()
        await consume(pending, generate, workers=32)
        atomic_json(out / ('timing-' + str(os.getpid()) + '.json'), {
            'completed_in_this_invocation': len(pending),
            'generation_wall_seconds': time.monotonic() - started,
            'generation_started_unix': generation_started_unix,
            'generation_finished_unix': time.time(),
        })
        print('INFERENCE_COMPLETE', variant['id'], flush=True)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    for key in ['model', 'variant', 'manifest', 'protocol', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--task-ids-file')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
