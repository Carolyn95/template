"""Model training script."""
from polyai_dataset.banking77 import Banking77
from polyai_dataset.clinc150 import Clinc150
from polyai_dataset.hwu64_sub import Hwu64Sub
from sentence_classifier import SentenceClassifier


DATASETS = {'bank': Banking77, 'clinc': Clinc150, 'hwu': Hwu64Sub}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', type=str, help='Huggingface pretrained model '
                        'or path to saved model on disk')
    parser.add_argument('dataset', type=str,
                        help=f'dataset to use {list(DATASETS)}')
    parser.add_argument('--out_dir', type=str, help='directory to save model')
    parser.add_argument('--batch', default=128,  # for 16 GB GPU RAM
                        help='per GPU training batch size')
    parser.add_argument('--lr', default=1e-4, help='learning rate')
    parser.add_argument('--epochs', default=10, help='training epochs')
    args = parser.parse_args()
    # Run
    model = SentenceClassifier.create(
        args.model, DATASETS[args.dataset].load(), int(args.batch))
    if args.out_dir:
        model.args.output_dir = args.out_dir
    else:
        suffix = f'{args.dataset}_{args.epochs}epochs'
        model.args.output_dir = f'{model.args.output_dir}_{suffix}'
    model.args.num_train_epochs = int(args.epochs)
    model.args.learning_rate = float(args.lr)
    model.train(test_dataset=model.data['test'])
