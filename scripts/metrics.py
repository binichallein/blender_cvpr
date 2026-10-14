"""BlenderGym photometric MSE and negative CLIP cosine distance, view-averaged."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPImageProcessor

class Metrics:
    def __init__(self, model_path, device='cpu'):
        torch.set_num_threads(4)
        self.device=device
        self.processor=CLIPImageProcessor.from_pretrained(model_path,local_files_only=True)
        self.model=CLIPModel.from_pretrained(model_path,local_files_only=True).to(device).eval()
        self.cache={}

    def features(self,path):
        path=Path(path).resolve()
        key=(str(path),path.stat().st_mtime_ns)
        if key not in self.cache:
            image=Image.open(path).convert('RGB')
            pixels=np.asarray(image,dtype=np.float32)/255.0
            inp=self.processor(images=image,return_tensors='pt').to(self.device)
            with torch.inference_mode():
                output=self.model.get_image_features(**inp)
                if not isinstance(output,torch.Tensor): output=output.pooler_output
                output=torch.nn.functional.normalize(output.float(),dim=-1)
            self.cache[key]=(pixels,output.cpu().numpy()[0])
        return self.cache[key]

    def pair(self,left,right):
        a,af=self.features(left);b,bf=self.features(right)
        if a.shape!=b.shape: raise ValueError(f'Shape mismatch {a.shape} != {b.shape}')
        return {'pl':float(np.mean((a-b)**2)),'nclip':float(max(0,1-np.dot(af,bf)))}

    def compare(self,left,right,views):
        rows=[{'view':v,**self.pair(Path(left)/v,Path(right)/v)} for v in views]
        return {k:float(np.mean([r[k] for r in rows])) for k in ('pl','nclip')} | {'per_view':rows}

def relative_error(prediction, start, floor):
    return {k:max(0.0,prediction[k]-floor[k])/max(start[k]-floor[k],1e-8) for k in ('pl','nclip')}

def passes(relative, rule):
    return all(relative[k]<=v for k,v in rule.items())

def should_expand(successes, completed, expected=50):
    return completed==expected and successes>=11
