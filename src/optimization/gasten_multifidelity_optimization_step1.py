import torch
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac import MultiFidelityFacade, Scenario
from torch.optim import Adam
import argparse
from dotenv import load_dotenv
import wandb
import os
from datetime import datetime
from smac.intensifier.hyperband import Hyperband

from src.utils.config import read_config
from src.gan import construct_gan, construct_loss
from src.datasets import load_dataset
from src.gan.update_g import UpdateGeneratorGAN
from src.metrics import fid
import math
from src.utils import MetricsLogger, group_images
from src.gan.train import train_disc, train_gen, loss_terms_to_str, evaluate
from src.utils.checkpoint import checkpoint_gan
from src.utils import load_z, set_seed, setup_reprod, create_checkpoint_path, gen_seed, seed_worker

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

    test_noise, test_noise_conf = load_z(config['test-noise'])
    batch_size = config['train']['step-1']['batch-size']

    fid_stats_mu, fid_stat_sigma = fid.load_statistics_from_path(config['fid-stats-path'])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), fid_stats_mu, fid_stat_sigma, device=device)

    fid_metrics = {
        'fid': original_fid
    }

    dataset_id = datetime.now().strftime("%b%dT%H-%M")

    fixed_noise = torch.randn(
        config['fixed-noise'], config["model"]["z_dim"], device=device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=config["num-workers"], worker_init_fn=seed_worker)

    log_every_g_iter = 50

    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='step-1-optimization',
               name=dataset_id)

    train_metrics = MetricsLogger(prefix='train')
    eval_metrics = MetricsLogger(prefix='eval')

    train_metrics.add('G_loss', iteration_metric=True)
    train_metrics.add('D_loss', iteration_metric=True)

    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(config, run_id)

    def train(params: Configuration, seed: int = 42, budget: int = 0) -> float:
        config['model']["architecture"]['g_num_blocks'] = params['n_blocks']
        config['model']["architecture"]['d_num_blocks'] = params['n_blocks']

        G, D = construct_gan(config["model"], img_size, device)

        g_crit, d_crit = construct_loss(config["model"]["loss"], D)
        g_updater = UpdateGeneratorGAN(g_crit)

        # Initialize optimizers
        g_opt = Adam(G.parameters(), lr=params['g_lr'], betas=(params['g_beta1'], params['g_beta2']))
        d_opt = Adam(D.parameters(), lr=params['d_lr'], betas=(params['d_beta1'], params['d_beta2']))

        train_state = {
            'epoch': 0,
            'best_epoch': 0,
            'best_epoch_metric': float('inf'),
        }

        for loss_term in g_updater.get_loss_terms():
            train_metrics.add(loss_term, iteration_metric=True)

        for loss_term in d_crit.get_loss_terms():
            train_metrics.add(loss_term, iteration_metric=True)

        for metric_name in fid_metrics.keys():
            eval_metrics.add(metric_name)

        eval_metrics.add_media_metric('samples')

        G.train()
        D.train()

        g_iters_per_epoch = int(math.floor(len(dataloader) / n_disc_iters))
        iters_per_epoch = g_iters_per_epoch * n_disc_iters

        for epoch in range(1, round(budget) + 1):
            data_iter = iter(dataloader)
            curr_g_iter = 0

            for i in range(1, iters_per_epoch + 1):
                data, _ = next(data_iter)
                real_data = data.to(device)

                ###
                # Update Discriminator
                ###
                d_loss, d_loss_terms = train_disc(
                    G, D, d_opt, d_crit, real_data, batch_size, train_metrics, device)

                ###
                # Update Generator
                # - update every 'n_disc_iterators' consecutive D updates
                ###
                if i % n_disc_iters == 0:
                    curr_g_iter += 1

                    g_loss, g_loss_terms = train_gen(g_updater,
                                                     G, D, g_opt, batch_size, train_metrics, device)

                    ###
                    # Log stats
                    ###
                    if curr_g_iter % log_every_g_iter == 0 or \
                            curr_g_iter == g_iters_per_epoch:
                        print('[%d/%d][%d/%d]\tG loss: %.4f %s; D loss: %.4f %s'
                              % (epoch, budget, curr_g_iter, g_iters_per_epoch, g_loss.item(),
                                 loss_terms_to_str(g_loss_terms), d_loss.item(),
                                 loss_terms_to_str(d_loss_terms)))

            ###
            # Sample images
            ###
            with torch.no_grad():
                G.eval()
                fake = G(fixed_noise).detach().cpu()
                G.train()

            img = group_images(fake, classifier=None, device=device)
            eval_metrics.log_image('samples', img)

            ###
            # Evaluate after epoch
            ###
            train_state['epoch'] += 1

            train_metrics.finalize_epoch()

            evaluate(G, fid_metrics, eval_metrics, batch_size,
                     test_noise, device, None)

            eval_metrics.finalize_epoch()

            config_checkpoint_dir = os.path.join(cp_dir, str(params.config_id))

            checkpoint_gan(
                G, D, g_opt, d_opt, train_state,
                {"eval": eval_metrics.stats, "train": train_metrics.stats}, config,
                epoch=epoch, output_dir=config_checkpoint_dir)

        return eval_metrics.stats['fid'][0]

    G_lr = Float("g_lr", (1e-4, 1e-3), default=0.0002)
    D_lr = Float("d_lr", (1e-4, 1e-3), default=0.0002)
    G_beta1 = Float("g_beta1", (0.1, 1), default=0.5)
    D_beta1 = Float("d_beta1", (0.1, 1), default=0.5)
    G_beta2 = Float("g_beta2", (0.1, 1), default=0.999)
    D_beta2 = Float("d_beta2", (0.1, 1), default=0.999)
    n_blocks = Integer("n_blocks", (1, 5), default=3)

    configspace = ConfigurationSpace()
    configspace.add_hyperparameters([G_lr, D_lr, G_beta1, D_beta1, G_beta2, D_beta2, n_blocks])

    scenario = Scenario(configspace, deterministic=True, n_trials=224, min_budget=2, max_budget=10)
    intensifier = Hyperband(scenario, eta=2)
    smac = MultiFidelityFacade(scenario, train, intensifier=intensifier)
    incumbent = smac.optimize()

    best_config = incumbent.get_dictionary()

    print("Best Configuration:", best_config)

    wandb.finish()

if __name__ == '__main__':
    main()
