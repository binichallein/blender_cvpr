import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_one import render

FAKE_BLENDER = """#!/usr/bin/env python
import json,sys
from pathlib import Path
args=sys.argv
out=Path(args[args.index('--output')+1])
script=Path(args[args.index('--script')+1])
counter=out/'counter'
counter.write_text(str(int(counter.read_text())+1 if counter.exists() else 1))
(out/'render1.png').write_bytes(b'fake-image')
(out/'render.json').write_text(json.dumps({'ok':True,'seconds':0}))
raise SystemExit(3 if 'FAIL' in script.read_text() else 0)
"""

class RenderCacheTests(unittest.TestCase):
    def test_changed_script_invalidates_cache_and_failed_exit_clears_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);blender=root/'blender';blender.write_text(FAKE_BLENDER);blender.chmod(0o755)
            script=root/'candidate.py';script.write_text('a=1')
            out=root/'renders'
            task={'base_blend':str(root/'base.blend'),'views':['render1.png']}
            def call(): return render(task,script,out,blender,root,'0',timeout=10)
            self.assertTrue(call()['ok']);self.assertEqual((out/'counter').read_text(),'1')
            self.assertTrue(call()['ok']);self.assertEqual((out/'counter').read_text(),'1')
            script.write_text('a=2')
            self.assertTrue(call()['ok']);self.assertEqual((out/'counter').read_text(),'2')
            script.write_text('FAIL')
            self.assertFalse(call()['ok'])
            self.assertFalse((out/'render.json').exists())
            self.assertTrue((out/'failure.json').exists())
if __name__=='__main__': unittest.main()
