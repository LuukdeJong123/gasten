import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_metric_learning import samplers

from datasets import get_mnist
from utils import binary_accuracy
from binary_dataset import BinaryDataset
from checkpoint_utils import checkpoint, load_checkpoint
from classifier import Classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', default='../data', help='Path to dataset')
    parser.add_argument('--out-dir', dest='out_dir', default='../out', help='Path to generated files')
    parser.add_argument('--name', dest='name', default=None, help='Name of the classifier for output files')
    parser.add_argument('--pos', dest='pos_class', default=7, type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=1, type=int, help='Negative class for binary classification')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=2)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--image-size', dest='image_size', type=int, default=28, help='Height / width of the images')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--early-stop', type=int, default=2, help='Early stopping criteria')
    parser.add_argument('--goal-acc-min', dest='goal_acc_min', type=float, default=None, help='Height / width of the images')
    parser.add_argument('--goal-acc-max', dest='goal_acc_max', type=float, default=None, help='Height / width of the images')

    return parser.parse_args()


def test(C, device, test_set, test_loader, criterion, verbose=True, cp_path=None):
    ###
    # Test
    ###
    if cp_path is not None:
        print("\nLoading checkpoint from best epoch for testing ...")
        load_checkpoint(cp_path, C, device=device)

    C.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    seq = tqdm(test_loader, desc='Test') if verbose else test_loader

    print("\n --- Test ---\n")
    for i, data in enumerate(seq, 0):
        X, y = data
        X = X.to(device)
        y = y.to(device)

        y_hat = C(X)
        loss = criterion(y_hat, y)

        running_accuracy += binary_accuracy(y_hat, y, avg=False).item()
        running_loss += loss.item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = running_accuracy / len(test_set)

    print("Test loss: {}".format(test_loss))
    print("Test accuracy: {}".format(test_accuracy))

    return {'acc': test_accuracy, 'loss': test_loss}


def main():
    args = parse_args()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    name = 'classifier.simple.{}v{}'.format(args.pos_class, args.neg_class) if args.name is None else args.name

    full_dataset = get_mnist(args.data_dir, args.image_size, train=True)
    dataset = BinaryDataset(full_dataset, args.pos_class, args.neg_class)

    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [int(5/6*len(dataset)), len(dataset) - int(5/6*len(dataset))])

    sampler = None
    if args.goal_acc_max is not None and args.goal_acc_min is not None:
        sampler_labels = train_set.dataset.targets[train_set.indices]
        sampler = samplers.MPerClassSampler(sampler_labels, args.batch_size / 2, batch_size=args.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=sampler, batch_size=args.batch_size, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    full_test_set = get_mnist(args.data_dir, args.image_size, train=False)
    test_set = BinaryDataset(full_test_set, args.pos_class, args.neg_class)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(torch.bincount(train_set.dataset.targets[train_set.indices].type(torch.uint8)))
    print(torch.bincount(test_set.targets.type(torch.uint8)))

    model_params = {
        'nc': 1,
        'nf': 1,
    }

    C = Classifier(model_params['nc'], model_params['nf']).to(device)
    print(C)
    opt = optim.Adam(C.parameters())
    criterion = nn.BCELoss()

    best_loss = float('inf')
    best_epoch = 0
    early_stop_tracker = 0
    early_stop = False

    for epoch in range(args.epochs):
        if early_stop:
            break
        print("\n --- Epoch {} ---\n".format(epoch+1), flush=True)

        ###
        # Train
        ###
        C.train()
        running_accuracy = 0.0
        running_loss = 0.0

        for i, data in enumerate(tqdm(train_loader, desc='Train'), 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()

            y_hat = C(X)
            loss = criterion(y_hat, y)
            loss.backward()

            opt.step()

            # Output training stats
            if args.goal_acc_max is not None and args.goal_acc_min is not None:
                print('[%d/%d][%d/%d]\tloss: %.4f\tAcc: %.4f'
                      % (epoch + 1, args.epochs, i + 1, len(train_loader), loss.item(),
                         binary_accuracy(y_hat, y, avg=True)))
                test_stats = test(C, device, test_set, test_loader, criterion, cp_path=None, verbose=False)
                if args.goal_acc_min <= test_stats['acc'] <= args.goal_acc_max:
                    name_with_acc = '{}.{}'.format(name, round(test_stats['acc'] * 100))
                    cp_path = checkpoint(C, name_with_acc, model_params, {'test': test_stats}, args,
                                         output_dir=args.out_dir)
                    print("\nSaved checkpoint to {}".format(cp_path))
                    exit(0)
                elif test_stats['acc'] > args.goal_acc_max:
                    name_with_acc = '{}.{}_fail'.format(name, round(test_stats['acc'] * 100))
                    cp_path = checkpoint(C, name_with_acc, model_params, {'test': test_stats}, args,
                                         output_dir=args.out_dir)
                    print("\nSaved checkpoint to {}".format(cp_path))
                    exit(-1)

                C.train()

            running_accuracy += binary_accuracy(y_hat, y, avg=False)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_set)

        print("Loss: {}".format(epoch_loss), flush=True)
        print("Accuracy: {}".format(epoch_accuracy), flush=True)

        ###
        # Validation
        ###
        C.eval()

        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(tqdm(val_loader, desc='Validation'), 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            y_hat = C(X)
            loss = criterion(y_hat, y)

            running_accuracy += binary_accuracy(y_hat, y, avg=False)
            running_loss += loss.item()

        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = running_accuracy / len(val_set)

        print("Loss: {}".format(epoch_loss), flush=True)
        print("Accuracy: {}".format(epoch_accuracy), flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            early_stop_tracker = 0

            cp_path = checkpoint(C, name, model_params, '', args, output_dir=args.out_dir)
            print("\nSaved checkpoint to {}".format(cp_path))
        else:
            if args.early_stop is not None:
                early_stop_tracker += 1
                print("\nEarly stop counter: {}/{}".format(
                    early_stop_tracker, args.early_stop))
                if early_stop_tracker == args.early_stop:
                    early_stop = True

    test(C, device, test_set, test_loader, criterion, cp_path=cp_path)


if __name__ == '__main__':
    main()
