"""End-to-end state-machine recovery with deterministic native tool responses."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

@unittest.skipUnless(importlib.util.find_spec('vllm'), 'Run in the vLLM inference environment')
class RecoveryIntegration(unittest.TestCase):
    def test_raw_completion_reused_and_earlier_attempt_selected(self):
        import infer_agentic as driver
        from transformers import AutoTokenizer
        from PIL import Image
        from agentic_sandbox import digest, atomic_json
        model='/limx_embap/tos/ckpt/mani/modelscope/models/Qwen/Qwen3.6-27B'
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        for thinking in [False,True]:
            with self.subTest(thinking=thinking), tempfile.TemporaryDirectory(prefix='bg-agentic-test-') as temp:
                root=Path(temp);taskdir=root/'fixture';taskdir.mkdir()
                (taskdir/'start.py').write_text('import bpy\n')
                for kind in ['start','goal']:
                    folder=taskdir/'renders'/kind;folder.mkdir(parents=True)
                    Image.new('RGB',(512,512),(255,0,0)).save(folder/'render1.png')
                task={'id':'fixture1','category':'lighting','directory':str(taskdir),
                      'views':['render1.png'],'base_blend':str(taskdir/'base.blend')}
                (root/'manifest.json').write_text(json.dumps({'all_tasks':[task]}))
                variant={'id':'fixture-mode','model':'Qwen/Qwen3.6-27B','thinking':thinking}
                protocol={'frozen':True,'max_new_tokens':65536,'context_capacity':98304,
                          'variants':[variant],'sampling':{'temperature':0.6,'seed':20261015}}
                (root/'protocol.json').write_text(json.dumps(protocol))
                config={'root':str(root),'protocol':str(root/'protocol.json'),
                        'manifest':str(root/'manifest.json'),'model':model}
                (root/'config.json').write_text(json.dumps(config))
                args=SimpleNamespace(config=str(root/'config.json'),variant='fixture-mode',task_ids=None,smoke_only=False)
                calls=[]
                class Engine:
                    async def generate(self,prompt,sampling_params,request_id):
                        calls.append(request_id)
                        if request_id.startswith('smoke'):
                            step=int(request_id.rsplit('-',1)[1])
                            name,param,value=('run_blender','code','print("tool-smoke")') if step==0 else ('submit','attempt_id','1')
                        else:
                            step=int(request_id.rsplit('-',1)[1])
                            name,param,value=('submit','attempt_id','1') if step==3 else (
                                'run_blender','code','print("first")' if step==1 else 'raise RuntimeError("second failed")')
                        text=(('Plan.</think>\n' if thinking else '')+
                              f'<tool_call>\n<function={name}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>')
                        ids=tokenizer.encode(text,add_special_tokens=False)
                        yield SimpleNamespace(finished=True,prompt_token_ids=[1]*100,
                            outputs=[SimpleNamespace(text=text,token_ids=ids,finish_reason='stop')])
                    def shutdown(self):pass
                executions=[]
                fail_once=[True]
                def execute(task,script,output,config):
                    if fail_once[0]:
                        fail_once[0]=False
                        raise RuntimeError('Injected infrastructure interruption after raw completion')
                    output=Path(output);output.mkdir(parents=True,exist_ok=True)
                    executions.append(Path(script).read_text())
                    ok='first' in Path(script).read_text()
                    result={'ok':ok,'returncode':0 if ok else 3,'timed_out':False,
                            'log_tail':'test execution failure' if not ok else '',
                            'script_sha256':digest(script),'seconds':0.01}
                    if ok:
                        shutil.copyfile(taskdir/'renders/start/render1.png',output/'render1.png')
                        atomic_json(output/'render.json',result)
                    return result
                with patch.object(driver.AsyncLLM,'from_engine_args',return_value=Engine()), \
                     patch.object(driver,'execute_candidate',side_effect=execute):
                    with self.assertRaises(ExceptionGroup) as interrupted:
                        asyncio.run(driver.run(args))
                    self.assertIn('Injected infrastructure', str(interrupted.exception.exceptions[0]))
                    shutil.move(root/'smoke',root/'smoke-before-resume')
                    asyncio.run(driver.run(args))
                self.assertEqual(calls.count('fixture1-1'),1,'Completed response was regenerated')
                self.assertEqual(len(executions),2)
                d=root/'predictions/fixture-mode/fixture1'
                g=json.loads((d/'generation.json').read_text())
                self.assertEqual(g['turns'],3)
                self.assertEqual(g['render_attempts'],2)
                self.assertEqual(g['selected_attempt'],1)
                self.assertTrue(g['explicit_submission'])
                self.assertEqual((d/'prediction.py').read_text(),'print("first")')
                total=sum(len(json.loads(p.read_text())['token_ids']) for p in d.glob('turn-*/completion.json'))
                self.assertEqual(g['output_tokens'],total)
                self.assertTrue((root/'renders/fixture-mode/fixture1/prediction/render1.png').exists())

if __name__=='__main__':
    unittest.main()
