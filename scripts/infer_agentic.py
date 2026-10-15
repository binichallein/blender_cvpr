"""Resumable vision/tool trajectories, continuously scheduled by vLLM AsyncLLM."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
import transformers
import vllm
from PIL import Image
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from agentic_core import (SYSTEM, TOOLS, TOTAL_TOKENS, TURN_TOKENS, MAX_TURNS,
    MAX_RENDERS, visible_history, split_native, parse_tool, validate_call,
    select_attempt, budget_text)
from agentic_sandbox import atomic_json, digest, gpu_lock, render
from async_queue import consume
from infer import build_messages
from infer_full import template_text


def initial_messages(task):
    messages = build_messages(task)
    messages[0]['content'] = [{'type': 'text', 'text': SYSTEM}]
    messages[1]['content'][-1] = {'type': 'text',
        'text': 'Use run_blender to try a complete revised script, inspect feedback, and call submit to finish.'}
    return messages

def prepare(messages, tokenizer, template, thinking):
    actual = visible_history(messages)
    prompt = tokenizer.apply_chat_template(actual, chat_template=template, tools=TOOLS,
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking, preserve_thinking=True)
    if prompt.rstrip().endswith('<think>') != thinking:
        raise RuntimeError('Native thinking prefix mismatch')
    images = []
    try:
        for message in actual:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if item['type'] == 'image':
                        with Image.open(item['url']) as image:
                            images.append(image.convert('RGB'))
        value = {'prompt': prompt}
        if images:
            value['multi_modal_data'] = {'image': images}
        return value, images, actual
    except BaseException:
        for image in images:
            image.close()
        raise

def execute_candidate(task, script, output, config):
    with gpu_lock(config['lock_root'], config['render_gpus']) as gpu:
        return render(task, script, output, config['blender'], config['infinigen'], gpu,
                      samples=512, seed=20261015, timeout=600)

def feedback(attempt, tokens, attempts, turns, views):
    result = attempt['result']
    if result['ok']:
        content = [{'type': 'text', 'text': f"attempt_id={attempt['attempt_id']}; execution and rendering succeeded. Compare these images to the original targets."}]
        for view in views:
            content += [{'type': 'text', 'text': f"Attempt {attempt['attempt_id']} view {view}:"},
                        {'type': 'image', 'url': str(Path(attempt['output']) / view)}]
    else:
        # Only bounded execution output. No score, reference image, or candidate ranking.
        content = [{'type': 'text', 'text':
            f"attempt_id={attempt['attempt_id']}; execution/rendering failed "
            f"(returncode={result['returncode']}, timeout={result['timed_out']}).\n"
            + result['log_tail']}]
    content += [{'type': 'text', 'text': budget_text(tokens, attempts, turns)}]
    return {'role': 'tool', 'content': content}

def finalize(task, state, destination, renders, protocol_sha):
    selected = select_attempt(state['attempts'], state.get('submitted'))
    target = Path(renders) / task['id'] / 'prediction'
    if selected:
        shutil.copyfile(selected['script'], destination / 'prediction.py')
        if selected['result']['ok']:
            target.mkdir(parents=True, exist_ok=True)
            for name in [*task['views'], 'render.json']:
                src = Path(selected['output']) / name
                if src.is_symlink():
                    raise RuntimeError('Refusing symlink in final render')
                shutil.copyfile(src, target / name)
    generation = {
        'task': task['id'], 'category': task['category'], 'ok': selected is not None,
        'protocol_sha256': protocol_sha, 'thinking': state['thinking'],
        'seconds': state['elapsed_seconds'], 'input_tokens': state['input_tokens'],
        'output_tokens': state['output_tokens'], 'reasoning_tokens': state['reasoning_tokens'],
        'answer_tokens': state['answer_tokens'], 'turns': state['turns'],
        'render_attempts': len(state['attempts']), 'selected_attempt': selected['attempt_id'] if selected else None,
        'explicit_submission': state.get('submitted') is not None,
        'stop_reason': state['stop_reason'], 'truncated_turns': state['truncated_turns'],
        'parse_errors': state['parse_errors'],
        'selection_policy': 'explicit submit; otherwise latest attempted script, including failures',
        'selected_script_sha256': digest(selected['script']) if selected else None,
        'truncated': state['stop_reason'] == 'token_budget',
    }
    if selected is None:
        generation['error'] = 'No candidate script was executed before termination'
    atomic_json(destination / 'generation.json', generation)
    print('TASK_COMPLETE', json.dumps(generation), flush=True)


async def run(args):
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text())
    protocol_path = Path(config['protocol'])
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = digest(protocol_path)
    assert protocol['frozen'] and protocol['max_new_tokens'] == TOTAL_TOKENS
    assert protocol.get('max_tokens_per_turn', TURN_TOKENS) == TURN_TOKENS
    assert protocol.get('max_render_attempts', MAX_RENDERS) == MAX_RENDERS
    assert protocol.get('max_assistant_turns', MAX_TURNS) == MAX_TURNS
    for filename, expected_hash in protocol.get('implementation_sha256', {}).items():
        assert digest(Path(__file__).with_name(filename)) == expected_hash, 'Frozen source changed: ' + filename
    variant = next(v for v in protocol['variants'] if v['id'] == args.variant)
    out = Path(config['root']) / 'predictions' / args.variant
    out.mkdir(parents=True, exist_ok=True)
    renders = Path(config['root']) / 'renders' / args.variant
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    template = template_text(config['model'])
    invocation_config = {
        'variant': variant, 'protocol_sha256': protocol_sha,
        'manifest_sha256': digest(config['manifest']),
        'model_config_sha256': digest(Path(config['model']) / 'config.json'),
        'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
        'backend': 'vllm.AsyncLLM', 'vllm': vllm.__version__,
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'dtype': 'bfloat16', 'context_capacity': protocol['context_capacity'],
        'sampling': protocol['sampling'], 'max_num_seqs': 8,
        'task_concurrency': 12, 'max_num_batched_tokens': 8192,
        'max_images': 15, 'max_image_pixels': 1048576,
        'system_prompt': SYSTEM, 'tools': TOOLS,
        'reasoning_parser': 'explicit think end, otherwise upstream PR 35687 implicit tool-call end',
        'tool_parser': 'vllm.Qwen3CoderToolParser non-streaming plus complete single-call validation',
        'code_sha256': {p: digest(Path(__file__).with_name(p)) for p in
                        ['infer_agentic.py', 'agentic_core.py', 'agentic_sandbox.py', 'blender_render.py']},
    }
    if (out / 'config.json').exists():
        assert json.loads((out / 'config.json').read_text()) == invocation_config, 'Mismatched resume'
    else:
        atomic_json(out / 'config.json', invocation_config)
    manifest = json.loads(Path(config['manifest']).read_text())
    tasks = manifest['all_tasks']
    if args.task_ids:
        requested = set(args.task_ids.split(','))
        tasks = [t for t in tasks if t['id'] in requested]
        assert len(tasks) == len(requested)
    pending = [t for t in tasks if not (out / t['id'] / 'generation.json').exists()]
    if not pending and not args.smoke_only:
        return
    atomic_json(out / f'invocation-{os.getpid()}.json', {
        'started_unix': time.time(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'task_ids': [t['id'] for t in pending], 'smoke_only': args.smoke_only})
    print('LOADING', args.variant, 'pending', len(pending), flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=config['model'], dtype='bfloat16', max_model_len=protocol['context_capacity'],
        max_num_seqs=8, gpu_memory_utilization=0.9, enforce_eager=True,
        max_num_batched_tokens=8192, tensor_parallel_size=1,
        limit_mm_per_prompt={'image': 15}, mm_processor_kwargs={'max_pixels': 1048576},
        seed=protocol['sampling']['seed'], enable_log_requests=False,
    ))

    async def request(messages, turn_dir, request_id, limit):
        prompt, images, actual = await asyncio.to_thread(prepare, messages, tokenizer, template, variant['thinking'])
        try:
            input_path = turn_dir / 'input.json'
            if input_path.exists():
                assert json.loads(input_path.read_text()) == actual, 'Changed resumed prompt'
            else:
                atomic_json(input_path, actual)
            complete = turn_dir / 'completion.json'
            if complete.exists():
                saved = json.loads(complete.read_text())
                assert saved['max_tokens'] == limit
                return saved
            started = time.monotonic()
            sampling = SamplingParams(**protocol['sampling'], max_tokens=limit,
                skip_special_tokens=False, output_kind=RequestOutputKind.FINAL_ONLY)
            result = None
            async for result in engine.generate(prompt=prompt, sampling_params=sampling, request_id=request_id):
                pass
            if result is None or not result.finished or len(result.outputs) != 1:
                raise RuntimeError('Incomplete inference request ' + request_id)
            answer = result.outputs[0]
            if answer.finish_reason not in ('stop', 'length'):
                raise RuntimeError('Infrastructure terminated request: ' + str(answer.finish_reason))
            completion = {'text': answer.text, 'token_ids': list(answer.token_ids),
                'input_tokens': len(result.prompt_token_ids), 'finish_reason': answer.finish_reason,
                'max_tokens': limit, 'seconds': time.monotonic() - started}
            atomic_json(complete, completion)
            return completion
        finally:
            for image in images:
                image.close()

    async def smoke():
        smoke_root = Path(config['root']) / 'smoke' / args.variant / str(os.getpid())
        smoke_root.mkdir(parents=True, exist_ok=True)
        image_path = smoke_root / 'red.png'
        Image.new('RGB', (128, 128), (255, 0, 0)).save(image_path)
        messages = [{'role': 'system', 'content': 'You are testing the two provided tools. Follow the user instructions exactly.'},
                    {'role': 'user', 'content': 'Call run_blender exactly once with code print("tool-smoke"). After its result, if the image is red call submit with attempt_id 1; if blue call submit with attempt_id 2.'}]
        rows = []
        for step in range(2):
            d = smoke_root / str(step); d.mkdir()
            c = await request(messages, d, f'smoke-{step}', 4096)
            assert c['finish_reason'] == 'stop'
            parts = split_native(c['token_ids'], tokenizer, variant['thinking'])
            call = parse_tool(parts['content'], tokenizer)
            assert call['name'] == ['run_blender', 'submit'][step], call
            if step == 1:
                assert call['arguments'] == {'attempt_id': 1}, call
            rows.append({'call': call, 'reasoning_tokens': parts['reasoning_tokens'],
                         'output_tokens': len(c['token_ids'])})
            messages.append({'role': 'assistant', 'reasoning_content': parts['reasoning'], 'content': parts['content']})
            messages.append({'role': 'tool', 'content': [
                {'type': 'text', 'text': 'attempt_id=1; execution succeeded. Here is the resulting image. Select according to the user instruction.'},
                {'type': 'image', 'url': str(image_path)}]})
        atomic_json(out / 'smoke.json', {'passed': True, 'variant': args.variant, 'rows': rows,
                                       'directory': str(smoke_root), 'protocol_sha256': protocol_sha})
        print('TOOL_VISION_SMOKE_PASSED', args.variant, flush=True)

    async def trajectory(task):
        destination = out / task['id']; destination.mkdir(exist_ok=True)
        state_path = destination / 'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            assert state['protocol_sha256'] == protocol_sha
        else:
            state = {'protocol_sha256': protocol_sha, 'thinking': variant['thinking'],
                     'messages': initial_messages(task), 'attempts': [], 'submitted': None,
                     'turns': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0,
                     'answer_tokens': 0, 'truncated_turns': 0, 'parse_errors': 0,
                     'elapsed_seconds': 0, 'stop_reason': None}
            atomic_json(destination / 'input.json', state['messages'])
            atomic_json(state_path, state)
        while state['stop_reason'] is None:
            if state['output_tokens'] >= TOTAL_TOKENS:
                state['stop_reason'] = 'token_budget'
                break
            if state['turns'] >= MAX_TURNS:
                state['stop_reason'] = 'turn_budget'
                break
            started = time.monotonic()
            turn_number = state['turns'] + 1
            turn_dir = destination / f'turn-{turn_number:03d}'; turn_dir.mkdir(exist_ok=True)
            c = await request(state['messages'], turn_dir,
                f"{task['id']}-{turn_number}", min(TURN_TOKENS, TOTAL_TOKENS - state['output_tokens']))
            state['turns'] += 1
            state['input_tokens'] += c['input_tokens']
            state['output_tokens'] += len(c['token_ids'])
            parts = None
            try:
                parts = split_native(c['token_ids'], tokenizer, variant['thinking'])
                state['reasoning_tokens'] += parts['reasoning_tokens']
                state['answer_tokens'] += parts['answer_tokens']
                state['messages'].append({'role': 'assistant',
                    'reasoning_content': parts['reasoning'], 'content': parts['content']})
            except ValueError as error:
                state['reasoning_tokens'] += len(c['token_ids']) if variant['thinking'] else 0
                state['answer_tokens'] += len(c['token_ids']) if not variant['thinking'] else 0
                # Keep interrupted reasoning visible on continuation; never execute it.
                state['messages'].append({'role': 'assistant',
                    'reasoning_content': c['text'] if variant['thinking'] else '',
                    'content': '' if variant['thinking'] else c['text']})
                parse_error = str(error)
            call = None
            if c['finish_reason'] == 'length':
                state['truncated_turns'] += 1
                parse_error = 'Response reached its token limit. No tool was executed. Use the remaining budget to issue a complete tool call.'
            elif parts is None:
                state['parse_errors'] += 1
            else:
                try:
                    call = parse_tool(parts['content'], tokenizer)
                    validate_call(call, state['attempts'])
                except (ValueError, KeyError, TypeError) as error:
                    state['parse_errors'] += 1
                    parse_error = str(error)
                    call = None
            if call is None:
                observation = {'role': 'tool', 'content': [{'type': 'text',
                    'text': 'Tool protocol feedback: ' + parse_error + '\n' +
                            budget_text(state['output_tokens'], state['attempts'], state['turns'])}]}
                state['messages'].append(observation)
                atomic_json(turn_dir / 'action.json', {'error': parse_error})
            elif call['name'] == 'submit':
                state['submitted'] = call['arguments']['attempt_id']
                state['stop_reason'] = 'submitted'
                atomic_json(turn_dir / 'action.json', call)
            else:
                attempt_id = len(state['attempts']) + 1
                d = destination / f'attempt-{attempt_id:03d}'; d.mkdir(exist_ok=True)
                script = d / 'candidate.py'
                code = call['arguments']['code']
                if script.exists():
                    assert script.read_text() == code, 'Candidate changed on resume'
                else:
                    script.write_text(code)
                print('RENDER_QUEUED', args.variant, task['id'], attempt_id, flush=True)
                result = await asyncio.to_thread(execute_candidate, task, script, d / 'output', config)
                attempt = {'attempt_id': attempt_id, 'script': str(script), 'output': str(d / 'output'),
                           'result': result, 'turn': turn_number}
                state['attempts'].append(attempt)
                state['messages'].append(feedback(attempt, state['output_tokens'], state['attempts'],
                                                   state['turns'], task['views']))
                atomic_json(turn_dir / 'action.json', {'call': call, 'attempt_id': attempt_id})
                print('RENDER_FINISHED', args.variant, task['id'], attempt_id, result['ok'], flush=True)
            state['elapsed_seconds'] += time.monotonic() - started
            atomic_json(state_path, state)
        atomic_json(state_path, state)
        finalize(task, state, destination, renders, protocol_sha)

    try:
        await smoke()
        if not args.smoke_only:
            await consume(pending, trajectory, workers=12)
            atomic_json(out / 'complete.json', {'tasks': len(tasks), 'finished_unix': time.time(),
                                               'protocol_sha256': protocol_sha})
            print('AGENTIC_COMPLETE', args.variant, flush=True)
    finally:
        engine.shutdown()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--task-ids')
    asyncio.run(run(p.parse_args()))
