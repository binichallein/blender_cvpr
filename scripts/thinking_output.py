"""Separate native reasoning from the final program; never execute reasoning code."""
from infer import extract_code


def split_output(token_ids, tokenizer, thinking):
    ids = list(token_ids)
    end_id = tokenizer.convert_tokens_to_ids('</think>')
    if thinking:
        if end_id not in ids:
            raise ValueError('Thinking did not terminate; no final answer')
        boundary = ids.index(end_id)
        reasoning_ids, final_ids = ids[:boundary], ids[boundary + 1:]
    else:
        # Some templates prefill an empty think block, outside generated tokens.
        # A generated nonempty block is a mode violation, not an off-mode answer.
        if end_id in ids:
            boundary = ids.index(end_id)
            prefix = tokenizer.decode(ids[:boundary], skip_special_tokens=True)
            if prefix.replace('<think>', '').strip():
                raise ValueError('Non-thinking request generated nonempty reasoning')
            reasoning_ids, final_ids = [], ids[boundary + 1:]
        else:
            reasoning_ids, final_ids = [], ids
    reasoning = tokenizer.decode(reasoning_ids, skip_special_tokens=True)
    final = tokenizer.decode(final_ids, skip_special_tokens=True)
    return {'reasoning': reasoning, 'final': final,
            'reasoning_tokens': len(reasoning_ids), 'answer_tokens': len(final_ids)}


def parse_program(token_ids, tokenizer, thinking, truncated=False):
    result = split_output(token_ids, tokenizer, thinking)
    if truncated:
        raise ValueError('Generation reached total token budget')
    result['code'] = extract_code(result['final'])
    return result
