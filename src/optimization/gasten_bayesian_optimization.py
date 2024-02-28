import torch
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario
from torch.optim import Adam
import argparse
from dotenv import load_dotenv
import wandb
import os
from datetime import datetime
import sys

from src.utils.config import read_config
from src.gan import construct_gan, construct_loss
from src.datasets import load_dataset
from src.utils import load_z
from src.gan.train import train as gan_train
from src.gan.update_g import UpdateGeneratorGAN
from src.metrics import fid


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

    pos_class = None
    neg_class = None
    if "binary" in config["dataset"]:
        pos_class = config["dataset"]["binary"]["pos"]
        neg_class = config["dataset"]["binary"]["neg"]

    dataset, num_classes, img_size = load_dataset(
        config["dataset"]["name"], config["data-dir"], pos_class, neg_class)

    n_disc_iters = config['train']['step-1']['disc-iters']

    dataset_id = datetime.now().strftime("%b%dT%H-%M")
    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='step-1-optimization',
               name=dataset_id)

    def train(params: Configuration, seed: int = 42) -> float:
        # config['model']["architecture"]['g_num_blocks'] = params['n_blocks']
        # config['model']["architecture"]['d_num_blocks'] = params['n_blocks']
        # config['model']["architecture"]['g_filter_dim'] = params['dimension']
        # config['model']["architecture"]['d_filter_dim'] = params['dimension']
        # config['model']["z_dim"] = params['dimension']

        Generator, Discriminator = construct_gan(config["model"], img_size, device)

        g_crit, d_crit = construct_loss(config["model"]["loss"], Discriminator)
        g_updater = UpdateGeneratorGAN(g_crit)

        # Initialize optimizers
        optimizer_G = Adam(Generator.parameters(), lr=params['g_lr'], betas=(params['g_beta1'], params['g_beta2']))
        optimizer_D = Adam(Discriminator.parameters(), lr=params['d_lr'], betas=(params['d_beta1'], params['d_beta2']))
        test_noise, test_noise_conf = load_z(config['test-noise-64'])
        batch_size = config['train']['step-1']['batch-size']
        n_epochs = config['train']['step-1']['epochs']
        # if params['dimension'] == 32:
        #     test_noise, test_noise_conf = load_z(config['test-noise-32'])
        # elif params['dimension'] == 64:
        #     test_noise, test_noise_conf = load_z(config['test-noise-64'])
        # elif params['dimension'] == 128:
        #     test_noise, test_noise_conf = load_z(config['test-noise-128'])
        # else:
        #     exit()

        mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
        fm_fn, dims = fid.get_inception_feature_map_fn(device)
        original_fid = fid.FID(
            fm_fn, dims, test_noise.size(0), mu, sigma, device=device)
        fid_metrics = {
            'fid': original_fid
        }

        train_state, best_cp, train_metrics, eval_metrics = gan_train(
            config, dataset, device, n_epochs, batch_size,
            Generator, optimizer_G, g_updater,
            Discriminator, optimizer_D, d_crit,
            test_noise, fid_metrics,
            n_disc_iters)

        current_score = eval_metrics.stats['fid'][0]

        return -current_score

    G_lr = Float("g_lr", (1e-5, 1e-1))
    D_lr = Float("d_lr", (1e-5, 1e-1))
    G_beta1 = Float("g_beta1", (0.1, 0.9))
    D_beta1 = Float("d_beta1", (0.1, 0.9))
    G_beta2 = Float("g_beta2", (0.1, 0.9))
    D_beta2 = Float("d_beta2", (0.1, 0.9))
    # n_blocks = Integer("n_blocks", (2, 4))
    # pre_training_epochs = Integer("epochs", (6, 10))
    # dimension = Categorical("dimension", [32, 64, 128])

    configspace = ConfigurationSpace()
    # configspace.add_hyperparameters([lr, beta1, beta2, n_blocks, pre_training_epochs, dimension])
    configspace.add_hyperparameters([G_lr, D_lr, G_beta1, D_beta1, G_beta2, D_beta2])

    scenario = Scenario(configspace, deterministic=True, n_trials=1)

    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()

    best_config = incumbent.get_dictionary()
    print("Best Configuration:", best_config)

    wandb.finish()

if __name__ == '__main__':
    main()
