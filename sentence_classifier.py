"""Sentence classification model."""
from pathlib import Path

from datasets import DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, set_seed,
    TrainingArguments, Trainer
)

from eval_accuracy import get_compute_metrics
from tokenise_data import tokenise_data


# Pretrained models from Huggingface
PRETRAINED = {
    'albert': 'albert-base-v1',
    'bart': 'facebook/bart-base',  # cased (half batch)
    'bert': 'bert-base-uncased',
    'deberta': 'microsoft/deberta-base',  # cased (half batch)
    'distilbert': 'distilbert-base-uncased',
    'distilroberta': 'distilroberta-base',  # cased
    'electra': 'google/electra-base-discriminator',
    'funnel': 'funnel-transformer/small',
    'mobilebert': 'google/mobilebert-uncased',
    'roberta': 'roberta-base',  # cased
    'squeezebert': 'squeezebert/squeezebert-uncased'
}


class SentenceClassifier():
    """Sentence classification model."""
    @staticmethod
    def create(model_name_or_path: str, dataset: DatasetDict,
               train_batch=128, seed: int=None):
        """Static factory method.

        Args:
            model_name_or_path: Huggingface pretrained model or path to saved
            model on disk
            dataset: dataset to use
            train_batch: training batch size
            seed: random seed to use
        """
        args = TrainingArguments(output_dir='', learning_rate=1e-4,
                                 per_device_train_batch_size=train_batch,
                                 evaluation_strategy='epoch',
                                 metric_for_best_model='accuracy',
                                 load_best_model_at_end=True,
                                 save_total_limit=1)
        if seed is not None:
            args.seed = seed
        args.output_dir = Path('models').joinpath(
            Path(model_name_or_path).name)
        set_seed(args.seed)  # affects initialisation of new classifier layer
        if model_name_or_path in PRETRAINED:
            model_name_or_path = PRETRAINED[model_name_or_path]
        tokeniser = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True)
        return SentenceClassifier(args, model_name_or_path, tokeniser,
                                  tokenise_data(tokeniser, dataset))

    def __init__(self, args, model_name_or_path, tokeniser, data):
        self.args = args
        self.model_name_or_path = model_name_or_path
        self.tokeniser = tokeniser
        self.data = data
        self.classes = data['train'].features['label'].names

    def train(self, train_dataset=None, eval_dataset=None, test_dataset=None):
        """Runs training, and evaluation if test dataset provided.

        If test dataset is provided, classification results are saved into
        'test_results.txt'.

        Args:
            train_dataset: tokenised training dataset ('train' split if None)
            eval_dataset: tokenised validation dataset ('validation' split if
            None)
            test_dataset: tokenised test dataset
        """
        if not train_dataset:
            train_dataset = self.data['train']
        if not eval_dataset:
            eval_dataset = self.data['validation']
        trainer = Trainer(
            args=self.args, tokenizer=self.tokeniser,
            model=AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, num_labels=len(self.classes)),
            compute_metrics=get_compute_metrics(self.classes),
            train_dataset=train_dataset, eval_dataset=eval_dataset
        )
        output_dir = Path(self.args.output_dir)
        trainer.train()
        trainer.save_model()
        trainer.state.save_to_json(output_dir.joinpath('trainer_state.json'))
        if test_dataset:
            self.eval(test_dataset, suffix='-train', trainer=trainer)

    def eval(self, test_dataset, suffix='', trainer=None,
             save_predictions=False):
        """Runs evaluation on tokenised test dataset.

        Classification results are saved into 'test_results.txt'. Predicted
        class labels are saved into 'test_predictions.txt'.

        Args:
            test_dataset: tokenised test dataset
            suffix: optional suffix to append to 'test_results' (eg. '-train'
            will save results into 'test_results-train.txt')
            trainer: trainer with model, metrics and data_collator provided
            save_predictions: saves predicted class labels
        """
        output_dir = Path(self.args.output_dir)
        if not trainer:
            trainer = Trainer(
                model=AutoModelForSequenceClassification.from_pretrained(
                    self.model_name_or_path, num_labels=len(self.classes)),
                compute_metrics=get_compute_metrics(
                    self.classes, save_predictions=save_predictions)
            )
        metrics = trainer.predict(test_dataset)[-1]
        print(f'\naccuracy = {metrics["eval_accuracy"]:.3f}')
        with open(output_dir.joinpath(f'test_results{suffix}.txt'),
                  'w') as writer:
            writer.write(f'{metrics["eval_report"]}\n')
            for key in ['accuracy', 'loss']:
                result = metrics[f'eval_{key}']
                writer.write(f'{key} = {result}\n')
        if save_predictions:
            with open(output_dir.joinpath('test_predictions.txt'),
                      'w') as writer:
                writer.write(' '.join(metrics['eval_predictions']) + '\n')
