"""Tests for polyai_dataset package."""
from pathlib import Path
import shutil
import unittest

from polyai_dataset.banking77 import Banking77
from polyai_dataset.clinc150 import Clinc150
from polyai_dataset.hwu64_sub import Hwu64Sub


class TestPolyAiDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_cache = Path('').cwd().joinpath('_cache')
        cls.datasets = [Banking77, Clinc150, Hwu64Sub]
        cls.data = {dataset.__name__: dataset.load()
                    for dataset in cls.datasets}  # slow

    @classmethod
    def tearDownClass(cls):
        if cls.fake_cache.exists():
            shutil.rmtree(cls.fake_cache)

    def test_load_invalid_data_dir(self):
        """Raises FileNotFoundException."""
        for dataset in TestPolyAiDataset.datasets:
            with self.subTest(dataset=dataset.__name__):
                with self.assertRaises(FileNotFoundError):
                    dataset.load('invalid/data_dir',
                                 cache_dir=TestPolyAiDataset.fake_cache)

    def test_load_loads_splits(self):
        """Verify that 'train', 'validation' and 'test' splits are loaded."""
        for dataset, data in TestPolyAiDataset.data.items():
            with self.subTest(dataset=dataset):
                for split in ['train', 'validation', 'test']:
                    with self.subTest(split=split):
                        self.assertIn(split, data.data)

    def test_load_loads_text(self):
        """Verify that 'text' field is loaded."""
        for dataset, data in TestPolyAiDataset.data.items():
            for name, split in data.data.items():
                with self.subTest(split=(dataset, name)):
                    self.assertIn('text', split.column_names)

    def test_load_loads_label(self):
        """Verify that 'label' field is loaded."""
        for dataset, data in TestPolyAiDataset.data.items():
            for name, split in data.data.items():
                with self.subTest(split=(dataset, name)):
                    self.assertIn('label', split.column_names)

    def test_load_splits_have_all_classes(self):
        for dataset, data in TestPolyAiDataset.data.items():
            with self.subTest(dataset=dataset):
                expected = {name: data[name].features['label'].num_classes
                            for name, split in data.items()}
                loaded = {name: len(set(split['label']))
                          for name, split in data.items()}
                self.assertEqual(expected, loaded)
