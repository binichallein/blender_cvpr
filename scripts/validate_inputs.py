"""Exercise the actual multimodal processors without running model inference."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from infer import build_messages

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['subset','all'],default='subset');a=p.parse_args();torch.set_num_threads(8)
    m=json.loads(Path(a.manifest).read_text());result={}
    tasks=[t for t in m['all_tasks'] if a.split=='all' or t['id'] in m['subset_ids']]
    for model in a.models:
        processor=AutoProcessor.from_pretrained(model,local_files_only=True);rows=[]
        for task in tasks:
            inputs=processor.apply_chat_template(build_messages(task),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            assert inputs['image_grid_thw'].shape[0]==2*len(task['views'])
            assert inputs['image_grid_thw'].shape[0]<=10
            rows.append({'task':task['id'],'input_tokens':inputs['input_ids'].shape[-1],'images':inputs['image_grid_thw'].shape[0]})
        result[Path(model).name]=rows
        print(Path(model).name,'validated',len(rows),'max_input_tokens',max(r['input_tokens'] for r in rows),flush=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
