import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from infer_replicas import partition_pending


class ReplicaPartitionTests(unittest.TestCase):
    def test_completed_answers_are_preserved_and_every_other_task_assigned_once(self):
        tasks = ['blendshape1', 'blendshape2', 'geometry1', 'lighting1', 'material1', 'placement1']
        groups = partition_pending(tasks, {'blendshape2', 'material1'}, 3)
        flattened = [task for group in groups for task in group]
        self.assertCountEqual(flattened, ['blendshape1', 'geometry1', 'lighting1', 'placement1'])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertNotIn('blendshape2', flattened)
        self.assertNotIn('material1', flattened)

    def test_more_replicas_than_remaining_tasks_and_complete_resume(self):
        self.assertEqual(partition_pending(['a', 'b'], {'a'}, 3), [['b'], [], []])
        self.assertEqual(partition_pending(['a'], {'a'}, 2), [[], []])
        with self.assertRaises(ValueError):
            partition_pending(['a'], set(), 0)
