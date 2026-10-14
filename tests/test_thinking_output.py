import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from thinking_output import parse_program, split_output


class Tokenizer:
    pieces = {
        1: 'Reasoning contains an unsafe draft:\n```python\nraise RuntimeError()\n```\n',
        2: '</think>', 3: '\n```python\nx = 7\n```', 4: '<think>\n',
        5: '\n', 6: 'still thinking', 7: '\n```python\nx=1\n```\n```python\nx=2\n```'
    }
    def convert_tokens_to_ids(self, text):
        return 2 if text == '</think>' else 999
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.pieces[i] for i in ids)


class ThinkingOutputTests(unittest.TestCase):
    def test_only_final_code_is_executed(self):
        result = parse_program([1, 2, 3], Tokenizer(), True)
        self.assertEqual(result['code'], 'x = 7\n')
        self.assertEqual(result['reasoning_tokens'], 1)
        self.assertNotIn('raise', result['code'])

    def test_reasoning_without_end_cannot_be_used_as_answer(self):
        with self.assertRaisesRegex(ValueError, 'did not terminate'):
            parse_program([1, 3], Tokenizer(), True)

    def test_off_mode_and_empty_block(self):
        self.assertEqual(parse_program([3], Tokenizer(), False)['code'], 'x = 7\n')
        self.assertEqual(parse_program([4, 5, 2, 3], Tokenizer(), False)['code'], 'x = 7\n')
        with self.assertRaisesRegex(ValueError, 'Non-thinking'):
            parse_program([1, 2, 3], Tokenizer(), False)

    def test_truncation_and_multiple_final_programs_fail(self):
        with self.assertRaisesRegex(ValueError, 'token budget'):
            parse_program([1, 2, 3], Tokenizer(), True, truncated=True)
        with self.assertRaises(ValueError):
            parse_program([1, 2, 7], Tokenizer(), True)
