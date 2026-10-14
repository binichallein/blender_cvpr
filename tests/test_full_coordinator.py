import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_full import matching_process, model_path, release_render_gpu


class CoordinatorTests(unittest.TestCase):
    def test_partial_download_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'Qwen3___6-27B'
            model.mkdir()
            for name in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'tokenizer.json']:
                (model / name).write_text('{}')
            (model / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors'}}))
            (model / 'part1.safetensors').write_bytes(b'weight')
            (model / 'part2.safetensors.tmp').write_bytes(b'incomplete')
            self.assertIsNone(model_path(root, 'Qwen/Qwen3.6-27B'))
            (model / 'part2.safetensors.tmp').rename(model / 'part2.safetensors')
            self.assertEqual(model_path(root, 'Qwen/Qwen3.6-27B'), model)

    def test_resume_rejects_reused_pid_or_different_output(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Path(directory) / '123'
            process.mkdir()
            (process / 'stat').write_text('123 (python) S')
            (process / 'cmdline').write_bytes(
                b'python\0/repo/scripts/infer_full.py\0--variant\0example\0--output\0/run/example\0')
            self.assertTrue(matching_process(123, 'example', '/run/example', directory))
            self.assertFalse(matching_process(123, 'other', '/run/example', directory))
            self.assertFalse(matching_process(123, 'example', '/run/other', directory))
            command = (process / 'cmdline').read_bytes().replace(b'/infer_full.py', b'/infer_replicas.py')
            (process / 'cmdline').write_bytes(command)
            self.assertTrue(matching_process(123, 'example', '/run/example', directory))
            (process / 'stat').write_text('123 (python) Z')
            self.assertFalse(matching_process(123, 'example', '/run/example', directory))


class RenderGpuReleaseTests(unittest.TestCase):
    def test_live_render_must_finish_before_gpu_is_available(self):
        finish = threading.Event()
        worker = threading.Thread(target=finish.wait)
        release = threading.Event()
        workers, releases = {'3': worker}, {'3': release}
        worker.start()
        try:
            self.assertFalse(release_render_gpu('3', workers, releases))
            self.assertTrue(release.is_set())
            self.assertIn('3', workers)
        finally:
            finish.set()
            worker.join()
        self.assertTrue(release_render_gpu('3', workers, releases))
        self.assertNotIn('3', workers)
        self.assertNotIn('3', releases)
        self.assertTrue(release_render_gpu('3', workers, releases))
