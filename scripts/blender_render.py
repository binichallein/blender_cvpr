"""Run inside Blender. Execute a script, restore fixed cameras, then render."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import bpy
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--script', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--views', nargs='+', default=['render1.png', 'render2.png'])
    ap.add_argument('--samples', type=int, default=512)
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--infinigen', required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.infinigen)
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    cameras = {o.name: (o.matrix_world.copy(), o.data.copy())
               for o in bpy.data.objects if o.type == 'CAMERA'}
    code = Path(args.script).read_text()
    namespace = {'__name__': '__main__', '__file__': args.script}
    exec(compile(code, args.script, 'exec'), namespace, namespace)
    # Cameras and render settings are part of evaluation, not editable answers.
    for name, (matrix, data) in cameras.items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'CAMERA':
            raise RuntimeError('Required camera removed: '+name)
        obj.matrix_world = matrix
        obj.data = data
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    gpu_count = 0
    for device in prefs.devices:
        device.use = device.type == 'CUDA'
        gpu_count += int(device.use)
    if not gpu_count:
        raise RuntimeError('No CUDA device available to Blender')
    scene.cycles.device = 'GPU'
    scene.cycles.seed = args.seed
    scene.cycles.use_animated_seed = False
    scene.cycles.samples = args.samples
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False
    rendered = []
    for view in args.views:
        camera_name = 'Camera'+Path(view).stem.removeprefix('render')
        camera = bpy.data.objects.get(camera_name)
        if camera is None or camera.type != 'CAMERA':
            raise RuntimeError('Missing evaluation camera: '+camera_name)
        scene.camera = camera
        scene.render.filepath = str(out/view)
        bpy.ops.render.render(write_still=True)
        if not (out/view).is_file():
            raise RuntimeError('Missing render: '+view)
        rendered.append(view)
    summary = {'ok': True, 'views': rendered, 'samples': args.samples,
               'seed': args.seed, 'blender_version': bpy.app.version_string,
               'seconds': time.monotonic()-started,
               'objects': [{'name': o.name, 'type': o.type,
                            'location': list(o.matrix_world.translation)}
                           for o in bpy.data.objects]}
    (out/'render.json').write_text(json.dumps(summary, indent=2))
    print('RENDER_COMPLETE', json.dumps({k:v for k,v in summary.items() if k!='objects'}), flush=True)

if __name__ == '__main__':
    main()
