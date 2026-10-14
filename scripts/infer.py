"""One-shot Qwen3-VL inference. No interpreter, tools, hidden goal code or repairs."""
import argparse
import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SYSTEM = """You edit Blender 3.6 Python programs to match target renders. You will receive paired current and target images and the complete current Python script. Infer the required scene change from the images. Return exactly one complete runnable Python script in a single fenced python block. Keep the supplied imports and API style. The base Blender scene is already loaded; your script is executed once in that scene. Do not load files, render images, change cameras, or change render/color settings. Do not use tools. Do not leave placeholders or omit unchanged code."""
CATEGORY = {
 'blendshape':'Adjust the named shape-key values to match the target shape/expression.',
 'geometry':'Edit the procedural geometry/node program to match the target geometry.',
 'lighting':'Adjust the existing lights to match target illumination and colors.',
 'material':'Edit the procedural material/node program to match the target appearance.',
 'placement':'Adjust the named objects to match their target positions; preserve the scene.'
}

def build_messages(task):
    d=Path(task['directory'])
    text=CATEGORY[task['category']]+"\nThe paired views use the same fixed cameras.\nCurrent Python script:\n\n\u0060\u0060\u0060python\n"+(d/'start.py').read_text()+"\n\u0060\u0060\u0060"
    content=[{'type':'text','text':text}]
    # Use all available corresponding views; do not expose task IDs or asset paths in text.
    for view in task['views']:
        content.extend([{'type':'text','text':f'Current view {view}:'},
                        {'type':'image','url':str(d/'renders/start'/view)},
                        {'type':'text','text':f'Target view {view}:'},
                        {'type':'image','url':str(d/'renders/goal'/view)}])
    content.append({'type':'text','text':'Return the complete modified Python script now.'})
    return [{'role':'system','content':SYSTEM},{'role':'user','content':content}]

def extract_code(raw):
    blocks=re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```',raw,re.S|re.I)
    code=blocks[0].strip() if len(blocks)==1 else raw.strip() if not blocks else ''
    if not code: raise ValueError('Expected exactly one Python program')
    ast.parse(code)
    return code+'\n'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['subset','all'],default='subset')
    p.add_argument('--max-new-tokens',type=int,default=8192)
    a=p.parse_args()
    protocol_path=Path(a.protocol)
    protocol=json.loads(protocol_path.read_text())
    assert protocol['frozen'], 'Freeze success criteria before benchmark inference'
    protocol_sha=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    m=json.loads(Path(a.manifest).read_text())
    ids=m['subset_ids'] if a.split=='subset' else [t['id'] for t in m['all_tasks']]
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(20261015)
    config={'model':Path(a.model).name,'model_config_sha256':hashlib.sha256((Path(a.model)/'config.json').read_bytes()).hexdigest(),
            'protocol_sha256':protocol_sha,'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            'dtype':'bfloat16','do_sample':False,'max_new_tokens':a.max_new_tokens,
            'transformers':transformers.__version__,'torch':torch.__version__,
            'system_prompt':SYSTEM,'category_instructions':CATEGORY,
            'input_views':'all matched released current/goal views','repairs':0,'attempts':1}
    config_path=out/'config.json'
    if config_path.exists(): assert json.loads(config_path.read_text())==config,'Refusing mismatched resume'
    else: config_path.write_text(json.dumps(config,indent=2))
    pending=[t for t in m['all_tasks'] if t['id'] in ids and not (out/t['id']/'generation.json').exists()]
    if not pending: return
    print('LOADING',Path(a.model).name,len(pending),flush=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map='auto',
             attn_implementation='sdpa',local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    print('MODEL_READY',flush=True)
    for task in pending:
        dest=out/task['id'];dest.mkdir(exist_ok=True)
        start=time.monotonic()
        row={'task':task['id'],'category':task['category'],'ok':False,'protocol_sha256':protocol_sha}
        try:
            messages=build_messages(task)
            (dest/'input.json').write_text(json.dumps(messages,indent=2))
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                       return_dict=True,return_tensors='pt').to(model.device)
            length=inputs['input_ids'].shape[-1]
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                                      use_cache=True)
            generated=output[0,length:]
            raw=processor.decode(generated,skip_special_tokens=True)
            (dest/'response.txt').write_text(raw)
            row.update(input_tokens=length,output_tokens=len(generated),truncated=len(generated)>=a.max_new_tokens)
            if row['truncated']: raise ValueError('Generation reached token budget')
            code=extract_code(raw)
            (dest/'prediction.py').write_text(code)
            row['ok']=True
            del inputs,output,generated
        except Exception as e:
            row['error']=repr(e)
            if isinstance(e,torch.cuda.OutOfMemoryError):
                # An infrastructure failure should stop rather than silently score all later tasks wrong.
                torch.cuda.empty_cache()
                raise
        row['seconds']=time.monotonic()-start
        temp=dest/'generation.json.tmp';temp.write_text(json.dumps(row,indent=2));temp.replace(dest/'generation.json')
        print(json.dumps(row),flush=True)
if __name__=='__main__': main()
