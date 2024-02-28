import torch
from torch.optim import Adam
from sklearn.model_selection import ParameterGrid
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

from src.utils.config import read_config
from src.gan import construct_gan, construct_loss
from src.datasets import load_dataset
from src.utils import load_z, set_seed, setup_reprod, create_checkpoint_path, gen_seed, seed_worker
from src.gan.train import train
from src.gan.update_g import UpdateGeneratorGAN
from src.metrics import fid, LossSecondTerm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        required=True, help="Config file")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    config = read_config(args.config_path)
    device = torch.device(config["device"])

    # Define hyperparameters and their possible values
    param_grid = {
        'lr': [0.0002, 0.0001],
        'beta1': [0.5, 0.9],
        'beta2': [0.5, 0.9],
        'batch_size': [32, 64],
        'n_blocks': [2, 3],
    }

    pos_class = None
    neg_class = None
    if "binary" in config["dataset"]:
        pos_class = config["dataset"]["binary"]["pos"]
        neg_class = config["dataset"]["binary"]["neg"]

    dataset, num_classes, img_size = load_dataset(
        config["dataset"]["name"], config["data-dir"], pos_class, neg_class)

    # Training loop with grid search for hyperparameter optimization
    best_score = float('-inf')
    best_params = None
    test_noise, test_noise_conf = load_z(config['test-noise'])
    mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), mu, sigma, device=device)
    fid_metrics = {
        'fid': original_fid
    }
    n_disc_iters = config['train']['step-1']['disc-iters']

    for params in tqdm(list(ParameterGrid(param_grid))):
        config['model']["architecture"]['g_num_blocks'] = params['n_blocks']
        config['model']["architecture"]['d_num_blocks'] = params['n_blocks']
        Generator, Discriminator = construct_gan(config["model"], img_size, device)

        g_crit, d_crit = construct_loss(config["model"]["loss"], Discriminator)
        g_updater = UpdateGeneratorGAN(g_crit)

        # Initialize optimizers
        optimizer_G = Adam(Generator.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
        optimizer_D = Adam(Discriminator.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

        # Training loop
        for epoch in range(1):
            train_state, best_cp, train_metrics, eval_metrics = train(
                config, dataset, device, 1, params['batch_size'],
                Generator, optimizer_G, g_updater,
                Discriminator, optimizer_D, d_crit,
                test_noise, fid_metrics,
                n_disc_iters)

        # Evaluate the model using a metric (you can use a validation set or another metric suitable for your task)
        current_score = eval_metrics.stats['fid'][0]

        # Check if the current set of hyperparameters is the best
        if current_score > best_score:
            best_score = current_score
            best_params = params

    print("Best Hyperparameters:", best_params)

if __name__ == '__main__':
    main()