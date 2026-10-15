import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_full_pipeline import verify_checkpoint


class CheckpointAuditTests(unittest.TestCase):
    def test_corrupt_same_size_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            path = directory / 'model.safetensors'
            original = b'valid weights'
            inventory = {'model': 'example', 'files': [{
                'Type': 'blob', 'Path': path.name, 'Size': len(original),
                'Sha256': hashlib.sha256(original).hexdigest(),
            }]}
            path.write_bytes(original)
            self.assertTrue(verify_checkpoint(directory, inventory)['passed'])
            path.write_bytes(b'wrong weights')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                verify_checkpoint(directory, inventory)

    def test_missing_and_traversal_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            item = {'Type': 'blob', 'Path': 'missing', 'Size': 1, 'Sha256': 'invalid'}
            with self.assertRaisesRegex(ValueError, 'Missing'):
                verify_checkpoint(directory, {'model': 'example', 'files': [item]})
            item['Path'] = '../outside'
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                verify_checkpoint(directory, {'model': 'example', 'files': [item]})
