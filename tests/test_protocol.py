import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from infer import extract_code, build_messages
from metrics import relative_error, passes, should_expand
from score import summarize

class ProtocolTests(unittest.TestCase):
    def test_code_extraction_and_invalid_response(self):
        self.assertEqual(extract_code("Answer:\n```python\nimport bpy\n```"),"import bpy\n")
        for response in ['```python\na=\n```','```python\na=1\n```\n```python\nb=2\n```']:
            with self.assertRaises((SyntaxError,ValueError)): extract_code(response)
    def test_full_run_gate_requires_all_fifty(self):
        self.assertFalse(should_expand(10,50))
        self.assertTrue(should_expand(11,50))
        self.assertFalse(should_expand(11,49))
    def test_noise_floor_and_noop(self):
        base={'pl':0.2,'nclip':0.1};floor={'pl':0.001,'nclip':0.002}
        self.assertEqual(relative_error(base,base,floor),{'pl':1.0,'nclip':1.0})
        self.assertEqual(relative_error(floor,base,floor),{'pl':0.0,'nclip':0.0})
        self.assertFalse(passes({'pl':1,'nclip':0.1},{'pl':0.2}))
    def test_data_errors_are_separate_from_model_failures(self):
        base={'pl':0.1,'nclip':0.2}
        rows=[{'task':'a','success':True,'metrics':base,'start_metrics':base},
              {'task':'b','success':False,'failure':'execution','start_metrics':base},
              {'task':'c','success':False,'invalid_data':'identical_program','start_metrics':base}]
        result=summarize(rows)
        self.assertEqual(result['tasks'],3)
        self.assertEqual(result['scorable_tasks'],2)
        self.assertEqual(result['success_rate'],0.5)

    def test_prompt_uses_start_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'start.py').write_text('import bpy\nx=123')
            (p/'goal.py').write_text('SECRET_REFERENCE_CODE')
            task={'category':'geometry','directory':d,'views':['render1.png','render2.png']}
            message=json.dumps(build_messages(task))
            self.assertNotIn('SECRET_REFERENCE_CODE',message)
            self.assertIn('x=123',message)
            images=[x for x in build_messages(task)[1]['content'] if x['type']=='image']
            self.assertEqual(len(images),4)
if __name__=='__main__': unittest.main()
