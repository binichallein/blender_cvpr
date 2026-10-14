"""Fetch the requested public checkpoint into a caller-selected shared cache."""
import argparse
from modelscope import snapshot_download
p=argparse.ArgumentParser()
p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
p.add_argument('--cache',required=True)
a=p.parse_args()
print(snapshot_download(a.model,cache_dir=a.cache,max_workers=4),flush=True)
