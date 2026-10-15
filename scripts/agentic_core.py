"""Frozen Blender tool interface and trajectory rules (no scoring imports)."""
import copy
import json

TOTAL_TOKENS = 65536
TURN_TOKENS = 32768
MAX_RENDERS = 5
MAX_TURNS = 12
SYSTEM = """You edit Blender 3.6 Python programs to match target renders. You receive paired current and target images and the complete current Python script. Infer the required scene change from the images. Keep the supplied imports and API style.

Use run_blender to execute a COMPLETE replacement Python script in a fresh copy of the original base scene. Each execution starts over: include unchanged code too. You receive execution errors or rendered images from the fixed evaluation cameras. Compare your rendered images with the target and revise as useful. Use submit with an earlier attempt_id to select your final answer, including an earlier version if a later revision was worse.

You have at most 5 run_blender calls (failed executions count), 12 assistant responses, and 65536 total generated tokens including reasoning and tool calls; each response allows at most 32768 tokens. Call one tool at a time. After the fifth execution, inspect its feedback and submit an attempt. On exhaustion without submission, your most recent attempted script is used, even if it failed.

Do not load files, inspect the filesystem, invoke shell commands, render images yourself, change cameras, or change render/color settings. The base Blender scene is already loaded. Only edit the scene through Blender Python APIs; the harness handles rendering. Do not access hidden target code or score data. No similarity scores are available. Your final answer must be selected using submit, not prose or a code fence."""
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'run_blender',
        'description': 'Execute a complete replacement Blender Python script in the original base scene and receive errors or all fixed-camera renders. Each call uses one of five attempts.',
        'parameters': {'type': 'object', 'properties': {
            'code': {'type': 'string', 'description': 'Complete runnable Python program, without markdown fences.'}},
            'required': ['code'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'submit',
        'description': 'Finish the task by selecting a previously executed attempt. Choose based on your own comparison with target images.',
        'parameters': {'type': 'object', 'properties': {
            'attempt_id': {'type': 'integer', 'description': 'Previously returned attempt_id (1 to 5).'}},
            'required': ['attempt_id'], 'additionalProperties': False}}}
]

def validate_call(call, attempts):
    name, arguments = call['name'], call['arguments']
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be an object')
    if name == 'run_blender':
        if set(arguments) != {'code'} or not isinstance(arguments['code'], str) or not arguments['code'].strip():
            raise ValueError('run_blender requires a nonempty code string only')
        if len(attempts) >= MAX_RENDERS:
            raise ValueError('Five execution attempts have been used. Call submit now.')
    elif name == 'submit':
        if set(arguments) != {'attempt_id'} or type(arguments['attempt_id']) is not int:
            raise ValueError('submit requires an integer attempt_id')
        if arguments['attempt_id'] not in [a['attempt_id'] for a in attempts]:
            raise ValueError('submit must select a previously executed attempt_id')
    else:
        raise ValueError('Unknown tool: ' + str(name))

def select_attempt(attempts, submitted=None):
    if submitted is not None:
        return next(a for a in attempts if a['attempt_id'] == submitted)
    # Deliberately do not choose the latest successful or lowest-error candidate.
    return attempts[-1] if attempts else None

def visible_history(messages):
    result = copy.deepcopy(messages)
    latest = max((i for i, m in enumerate(result)
                  if m['role'] == 'tool' and isinstance(m['content'], list)
                  and any(x.get('type') == 'image' for x in m['content'])), default=-1)
    for i, message in enumerate(result):
        if message['role'] == 'tool' and i != latest and isinstance(message['content'], list):
            message['content'] = [
                {'type': 'text', 'text': '[Earlier render image omitted; its textual result and code remain in history.]'}
                if x.get('type') == 'image' else x for x in message['content']]
    return result

def split_native(ids, tokenizer, thinking):
    ids = list(ids)
    end = tokenizer.convert_tokens_to_ids('</think>')
    tool = tokenizer.convert_tokens_to_ids('<tool_call>')
    boundary, implicit = None, False
    if thinking:
        if end in ids:
            boundary = ids.index(end)
            reason_ids, final_ids = ids[:boundary], ids[boundary + 1:]
        elif tool in ids:
            # Matches upstream vLLM PR 35687: a native tool call ends reasoning.
            boundary, implicit = ids.index(tool), True
            reason_ids, final_ids = ids[:boundary], ids[boundary:]
        else:
            raise ValueError('Reasoning did not terminate or issue a tool call')
    elif end in ids:
        boundary = ids.index(end)
        prefix = tokenizer.decode(ids[:boundary], skip_special_tokens=False).replace('<think>', '').strip()
        if prefix:
            raise ValueError('Non-thinking request generated nonempty reasoning')
        reason_ids, final_ids = [], ids[boundary + 1:]
    else:
        reason_ids, final_ids = [], ids
    clean = lambda values: tokenizer.decode(values, skip_special_tokens=False).replace('<|im_end|>', '').replace('<|endoftext|>', '')
    return {'reasoning': clean(reason_ids).removeprefix('<think>').strip(),
            'content': clean(final_ids).strip(), 'reasoning_tokens': len(reason_ids),
            'answer_tokens': len(final_ids), 'implicit_reasoning_end': implicit}

def parse_tool(content, tokenizer):
    # Use vLLM's official non-streaming Qwen XML parser, with a strict complete-call gate.
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
    if content.count('<tool_call>') != 1 or content.count('</tool_call>') != 1:
        raise ValueError('Return exactly one complete native tool call')
    if content.split('</tool_call>', 1)[1].strip():
        raise ValueError('Unexpected content after tool call')
    request = ChatCompletionRequest(model='Qwen/Qwen3.6-27B',
        messages=[{'role': 'user', 'content': 'Use the provided tools.'}], tools=TOOLS)
    parsed = Qwen3CoderToolParser(tokenizer).extract_tool_calls(content, request)
    if not parsed.tools_called or len(parsed.tool_calls) != 1:
        raise ValueError('Could not parse exactly one tool call')
    call = parsed.tool_calls[0]
    return {'name': call.function.name, 'arguments': json.loads(call.function.arguments),
            'content': parsed.content or ''}

def budget_text(tokens, attempts, turns):
    return (f'Remaining output token budget: {max(0, TOTAL_TOKENS-tokens)}. '
            f'Execution attempts used: {len(attempts)}/{MAX_RENDERS}. '
            f'Assistant responses used: {turns}/{MAX_TURNS}. '
            + ('No execution attempts remain; select an attempt with submit.' if len(attempts) >= MAX_RENDERS else ''))
