import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from agentic_core import validate_call, select_attempt, visible_history, split_native, MAX_RENDERS

class Tokens:
    values = {1:'reason',2:'</think>',3:'<tool_call>',4:'call',5:'</tool_call>',6:'<think>',7:'<|im_end|>'}
    def convert_tokens_to_ids(self, x):
        return {v:k for k,v in self.values.items()}.get(x)
    def decode(self, ids, **kwargs):
        return ''.join(self.values[i] for i in ids)

class AgenticTests(unittest.TestCase):
    def test_final_fallback_does_not_choose_successful_candidate(self):
        attempts = [{'attempt_id':1,'ok':True},{'attempt_id':2,'ok':False}]
        self.assertEqual(select_attempt(attempts)['attempt_id'],2)
        self.assertEqual(select_attempt(attempts,1)['attempt_id'],1)
        self.assertIsNone(select_attempt([]))
    def test_reject_unexecuted_submission(self):
        with self.assertRaises(ValueError):
            validate_call({'name':'submit','arguments':{'attempt_id':2}},[{'attempt_id':1}])
    def test_failed_attempts_still_exhaust_budget(self):
        attempts = [{'attempt_id':i,'ok':False} for i in range(1,MAX_RENDERS+1)]
        with self.assertRaises(ValueError):
            validate_call({'name':'run_blender','arguments':{'code':'print(1)'}},attempts)
        validate_call({'name':'submit','arguments':{'attempt_id':1}},attempts)
    def test_no_boolean_or_extra_argument_submission(self):
        for args in [{'attempt_id':True},{'attempt_id':1,'score':0}]:
            with self.assertRaises(ValueError):
                validate_call({'name':'submit','arguments':args},[{'attempt_id':1}])
    def test_only_latest_tool_images_retained_without_mutating_archive(self):
        image = lambda p: {'type':'image','url':p}
        original = [{'role':'user','content':[image('start'),image('goal')]},
                    {'role':'tool','content':[image('old'),{'type':'text','text':'old result'}]},
                    {'role':'assistant','content':'code'},
                    {'role':'tool','content':[image('new')]}]
        actual = visible_history(original)
        self.assertEqual(actual[0],original[0])
        self.assertEqual(actual[3],original[3])
        self.assertEqual(actual[1]['content'][0]['type'],'text')
        self.assertEqual(original[1]['content'][0]['type'],'image')
    def test_explicit_reasoning_end(self):
        result = split_native([1,2,3,4,5,7],Tokens(),True)
        self.assertEqual(result['reasoning'],'reason')
        self.assertEqual(result['content'],'<tool_call>call</tool_call>')
        self.assertEqual(result['reasoning_tokens'],1)
    def test_native_tool_boundary_ends_reasoning(self):
        result = split_native([1,3,4,5,7],Tokens(),True)
        self.assertTrue(result['implicit_reasoning_end'])
        self.assertEqual(result['content'],'<tool_call>call</tool_call>')
    def test_unterminated_reasoning_is_not_executed(self):
        with self.assertRaises(ValueError):
            split_native([1],Tokens(),True)
    def test_off_mode_rejects_nonempty_thinking(self):
        with self.assertRaises(ValueError):
            split_native([6,1,2,3,4,5],Tokens(),False)
        self.assertEqual(split_native([3,4,5],Tokens(),False)['reasoning_tokens'],0)

if __name__=='__main__':
    unittest.main()
