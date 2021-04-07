"""Tests for tokenise_data."""
import unittest

from datasets import Dataset
from transformers import AutoTokenizer

from tokenise_data import tokenise_data


class TestTokeniseData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokeniser = AutoTokenizer.from_pretrained(
            'roberta-base', use_fast=True,
            add_prefix_space=True)  # use a cased model

    def setUp(self):
        self.data = Dataset.from_dict(
            {'text': ['foo bar', 'aa bb c def'], 'label': [0, 3]})

    def test_tokenise_data_no_text_field(self):
        """Raises KeyError."""
        no_tokens = Dataset.from_dict({'label': [1]})
        with self.assertRaises(KeyError):
            tokenise_data(TestTokeniseData.tokeniser, no_tokens)

    def test_tokenise_data_no_label_field(self):
        """Raises KeyError."""
        no_tags = Dataset.from_dict({'text': ['foo bar']})
        with self.assertRaises(KeyError):
            tokenise_data(TestTokeniseData.tokeniser, no_tags)

    def test_tokenise_data_adds_input_ids_field(self):
        # Verify that it's not already present
        self.assertNotIn('input_ids', self.data.features)
        # Test
        tokenised = tokenise_data(TestTokeniseData.tokeniser, self.data)
        self.assertIn('input_ids', tokenised.features)

    def test_tokenise_data_uncased_converts_text_to_lowercase(self):
        mixed_case = 'MiXeD CaSe tExT'
        tokeniser = TestTokeniseData.tokeniser
        # Verify that input and tokeniser is mixed case
        self.assertFalse(mixed_case.islower())
        self.assertEqual(mixed_case, tokeniser.decode(
            tokeniser(mixed_case).input_ids, skip_special_tokens=True).strip())
        # Test
        data = Dataset.from_dict({'text': [mixed_case],
                                  'label': [1]})
        tokenised = tokenise_data(tokeniser, data=data, uncased=True)
        converted = tokeniser.decode(
            tokenised['input_ids'][0], skip_special_tokens=True).strip()
        self.assertEqual(mixed_case.lower(), converted)
