import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_full import matching_process, model_path, release_render_gpu, replica_owned_gpus, inference_order


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

    def test_replica_gpu_is_released_only_after_its_child_exits(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = output / 'replica-runs/100/run.json'
            allocated = ['3', '0']
            self.assertEqual(replica_owned_gpus(100, 'example', output, allocated), allocated)
            record.parent.mkdir(parents=True)
            metadata = {'variant': 'example', 'gpu_ids': allocated,
                        'children': [{'pid': 101, 'gpu': '3'}, {'pid': 102, 'gpu': '0'}]}
            record.write_text(json.dumps(metadata))
            with patch('run_full.matching_process', return_value=True):
                self.assertEqual(replica_owned_gpus(100, 'example', output, allocated), allocated)
            with patch('run_full.matching_process', side_effect=lambda pid, *args: pid == 102):
                self.assertEqual(replica_owned_gpus(100, 'example', output, allocated), ['0'])
            with patch('run_full.matching_process', return_value=False):
                self.assertEqual(replica_owned_gpus(100, 'example', output, allocated), [])

    def test_partial_replica_metadata_never_releases_a_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = output / 'replica-runs/100/run.json'
            record.parent.mkdir(parents=True)
            for text in ['{', json.dumps({'variant': 'example', 'gpu_ids': ['3', '0'],
                                        'children': [{'pid': 101, 'gpu': '3'}]})]:
                record.write_text(text)
                self.assertEqual(replica_owned_gpus(100, 'example', output, ['3', '0']), ['3', '0'])

    def test_released_replica_gpu_is_never_reacquired_by_stale_metadata(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = output / 'replica-runs/100/run.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({'variant': 'example', 'gpu_ids': ['3', '0'],
                'children': [{'pid': 101, 'gpu': '3'}, {'pid': 102, 'gpu': '0'}]}))
            with patch('run_full.matching_process', return_value=True):
                self.assertEqual(replica_owned_gpus(100, 'example', output, ['0']), ['0'])


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


class ProtocolVariantsTests(unittest.TestCase):
    def test_new_pair_is_scheduled_once_without_unrequested_models(self):
        variants = {'qwen3.5-122b-nothink': {}, 'qwen3.5-122b-think': {}}
        self.assertEqual(inference_order(variants), list(variants))

    def test_mixed_protocol_preserves_original_priority_and_all_new_variants(self):
        variants = {'qwen3.5-122b-think': {}, 'qwen3-vl-8b-think': {},
                    'qwen3-vl-32b-nothink': {}, 'qwen3.5-122b-nothink': {}}
        self.assertEqual(inference_order(variants), ['qwen3-vl-32b-nothink',
            'qwen3-vl-8b-think', 'qwen3.5-122b-think', 'qwen3.5-122b-nothink'])
