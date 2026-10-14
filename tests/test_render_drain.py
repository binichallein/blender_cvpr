import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_model import inference_finished
from render_one import render

class RenderDrainTests(unittest.TestCase):
    def test_generation_exit_does_not_skip_late_render_work(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a').mkdir();(root/'a/generation.json').write_text('{}')
            tasks=[{'id':'a'}]
            self.assertFalse(inference_finished(0,set(),tasks,root))
            self.assertTrue(inference_finished(0,{'a'},tasks,root))
            with self.assertRaises(RuntimeError):
                inference_finished(0,set(),tasks+[{'id':'missing'}],root)
            with self.assertRaises(RuntimeError):
                inference_finished(1,set(),tasks,root)

    def test_failed_model_execution_is_not_retried_on_resume(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/'candidate.py';script.write_text('raise ValueError()')
            output=root/'output';output.mkdir()
            renderer=Path(__file__).resolve().parents[1]/'scripts/blender_render.py'
            expected={'ok':False,'returncode':3,'views':['render1.png'],'samples':512,
                      'seed':20261015,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
                      'runner_sha256':hashlib.sha256(renderer.read_bytes()).hexdigest(),
                      'base_blend':'test.blend'}
            (output/'failure.json').write_text(json.dumps(expected))
            with patch('render_one.subprocess.Popen',side_effect=AssertionError('must not execute twice')):
                result=render({'views':['render1.png'],'base_blend':'test.blend'},script,output,
                              root/'blender',root/'infinigen','0',reuse_failures=True)
            self.assertEqual(result,expected)
