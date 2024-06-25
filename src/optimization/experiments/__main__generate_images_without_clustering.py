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


    with open(f'{os.environ["FILESDIR"]}/step-2-best-config-{args.dataset}-{pos_class}v{neg_class}.txt', 'r') as file:
        lines = file.read().splitlines()
        gan_path = lines[0]
        best_config_optim = json.loads(lines[1].replace("'", '"'))

    config_optim = read_config(args.config_path_optim)
    config_optim["project"] = f"{config_optim['project']}-{pos_class}v{neg_class}"
    config_clustering = read_config_clustering(args.config_path_clustering)

    config_clustering['dataset']['name'] = args.dataset
    config_clustering['dataset']['binary']['pos'] = pos_class
    config_clustering['dataset']['binary']['neg'] = neg_class
    config_clustering['dir']['fid-stats'] = fid_stats_path
    config_clustering['gasten']['gan_path'] = gan_path
    config_clustering['gasten']['weight'] = best_config_optim['weight']
    gan_path_splitted = gan_path.split('/')
    config_clustering['gasten']['run-id'] = gan_path_splitted[len(gan_path_splitted) - 2]
    config_clustering["project"] = f"{config_clustering['project']}-{pos_class}v{neg_class}"

    netG, C, C_emb, classifier_name = load_gasten(config_clustering, best_config_optim['classifier'], best_config_optim)

    device = config_clustering["device"]
    batch_size = config_clustering['batch-size']

    config_run = {
        'step': 'image_generation',
        'classifier_name': classifier_name,
        'gasten': {
            'epoch1': config_clustering['gasten']['epoch']['step-1'],
            'epoch2': config_clustering['gasten']['epoch']['step-2'],
            'weight': config_clustering['gasten']['weight']
        },
        'probabilities': {
            'min': 0.5 - config_clustering['clustering']['acd'],
            'max': 0.5 + config_clustering['clustering']['acd']
        },
        'generated_images': config_clustering['clustering']['fixed-noise']
    }

    # Initialize variables
    images_array = []
    syn_images_f = torch.empty(0)  # Initialize empty tensor

    while len(syn_images_f) < 1751:
        test_noise = torch.randn(config_run['generated_images'], config_clustering["clustering"]["z-dim"],
                                 device=device)
        noise_loader = DataLoader(TensorDataset(test_noise), batch_size=batch_size, shuffle=False)

        images_array.clear()

        for idx, batch in enumerate(tqdm(noise_loader, desc='Evaluating fake images')):
            with torch.no_grad():
                netG.eval()
                batch_images = netG(*batch)

            images_array.append(batch_images)

        images = torch.cat(images_array, dim=0)

        with torch.no_grad():
            pred = C(images).cpu().detach().numpy()

        mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
        filtered_images = images[mask]

        syn_images_f = torch.cat([syn_images_f.to(device), filtered_images.to(device)], dim=0)

    shuffled_indices = torch.randperm(syn_images_f.size(0))
    syn_images_f_shuffled = syn_images_f[shuffled_indices]

    sampled_images = syn_images_f_shuffled[:1750]

    #Save
    save(config_optim, sampled_images, f"{args.dataset}_{pos_class}v{neg_class}")


if __name__ == '__main__':
    main()
