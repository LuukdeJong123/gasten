import itertools
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import json
import torch

from src.utils import create_and_store_z, gen_seed, set_seed
from dotenv import load_dotenv
from src.utils.config import read_config_clustering
from src.clustering.generate_embeddings import load_gasten
from src.utils.config import read_config

load_dotenv()
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', dest='dataroot',
                    default=f"{os.environ['FILESDIR']}/data", help='Dir with dataset')
parser.add_argument('--out-dir-models', dest='out_dir_models',
                    default=f"{os.environ['FILESDIR']}/models", help='Path to generated files')
parser.add_argument('--out-dir-data', dest='out_dir_data',
                    default=f"{os.environ['FILESDIR']}/data/z", help='Path to generated files')
parser.add_argument('--dataset', dest='dataset',
                    default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
parser.add_argument('--n-classes', dest='n_classes',
                    default=10, help='Number of classes in dataset')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--batch-size', dest='batch_size',
                    type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='ADAM opt learning rate')

parser.add_argument('--pos', dest='pos_class', default=9,
                    type=int, help='Positive class for binary classification')
parser.add_argument('--neg', dest='neg_class', default=4,
                    type=int, help='Negative class for binary classification')
parser.add_argument('--epochs', type=str, default="1",
                    help='List of number of epochs to train for')
parser.add_argument('--classifier-type', dest='clf_type',
                    type=str, help='list with elements "cnn" or "mlp"', default='cnn')
parser.add_argument('--nf', type=str, default="1,2,4,8",
                    help='List of possible num features')
parser.add_argument("--nz", dest="nz", default=2000, type=int)
parser.add_argument("--z-dim", dest="z_dim", default=64, type=int)
parser.add_argument("--config", dest="config_path_optim", required=True, help="Config file gasten")
parser.add_argument("--config_clustering", dest="config_path_clustering", required=True, help="Config file clustering")
parser.add_argument("--seed", type=int, default=None)


def save(config, images, name):
    print("> Save ...")
    """
    save images 
    """
    path = os.path.join(config['out-dir'], 'images')
    torch.save(images, f"{path}/images_{name}.pt")


def main():
    args = parser.parse_args()

    seed = gen_seed() if args.seed is None else args.seed

    set_seed(seed)

    l_epochs = list(set([e
                         for e in args.epochs.split(",") if e.isdigit()]))
    l_clf_type = list(set([ct
                           for ct in args.clf_type.split(",")]))
    l_nf = list(set([nf
                     for nf in args.nf.split(",") if nf.isdigit()]))
    l_epochs.sort()
    l_clf_type.sort()
    l_nf.sort()

    if args.pos_class is not None and args.neg_class is not None:
        pos_class = str(args.pos_class)
        neg_class = str(args.neg_class)
    else:
        print('No positive and or negative class given!')
        exit()

    fid_stats_path = f"{os.environ['FILESDIR']}/data/fid-stats/stats.inception.{args.dataset}.{pos_class}v{neg_class}.npz"
    if not os.path.exists(fid_stats_path):
        print(f"\nGenerating FID score for {pos_class}v{neg_class} ...")
        subprocess.run(['python3', '-m', 'src.metrics.fid',
                        '--data', args.dataroot,
                        '--dataset', args.dataset,
                        '--device', args.device,
                        '--pos', pos_class, '--neg', neg_class])

    if not os.path.exists(f"{os.environ['FILESDIR']}/models/{args.dataset}.{pos_class}v{neg_class}"):
        print(f"\nGenerating classifiers for {pos_class}v{neg_class} ...")
        for clf_type, nf, epochs in itertools.product(l_clf_type, l_nf, l_epochs):
            print("\n", clf_type, nf, epochs)
            proc = subprocess.run(["python3", "-m", "src.classifier.train",
                                   "--device", args.device,
                                   "--data-dir", args.dataroot,
                                   "--out-dir", args.out_dir_models,
                                   "--dataset", args.dataset,
                                   "--pos", pos_class,
                                   "--neg", neg_class,
                                   "--classifier-type", clf_type,
                                   "--nf", nf,
                                   "--epochs", epochs,
                                   "--batch-size", str(args.batch_size),
                                   "--lr", str(args.lr)],
                                  capture_output=True)
            for line in proc.stdout.split(b'\n')[-4:-1]:
                print(line.decode())

    if not os.path.exists(f"{os.environ['FILESDIR']}/data/z/z_{args.nz}_{args.z_dim}"):
        create_and_store_z(
            args.out_dir_data, args.nz, args.z_dim,
            config={'seed': seed, 'n_z': args.nz, 'z_dim': args.z_dim})

    subprocess.run(['python3', '-m', 'src.optimization.gasten_multifidelity_optimization_step1',
                    '--config', args.config_path_optim, '--pos', pos_class, '--neg', neg_class,
                    '--dataset', args.dataset, '--fid-stats', fid_stats_path])

    classifiers = os.listdir(os.path.join(os.environ['FILESDIR'], 'models', f"{args.dataset}.{pos_class}v{neg_class}"))
    classifier_paths = ",".join(
        [f"{os.environ['FILESDIR']}/models/{args.dataset}.{pos_class}v{neg_class}/{classifier}" for classifier in
         classifiers])

    subprocess.run(['python3', '-m', 'src.optimization.gasten_multifidelity_optimization_step2',
                    '--config', args.config_path_optim, '--classifiers', classifier_paths, '--pos', pos_class,
                    '--neg', neg_class, '--dataset', args.dataset, '--fid-stats', fid_stats_path])


if __name__ == '__main__':
    main()
