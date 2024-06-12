import torch
from torch.optim import Adam
import argparse
from dotenv import load_dotenv
import wandb
import math
import os
from scipy.stats import uniform, randint, loguniform
import time

from src.utils.config import read_config
from src.gan import construct_gan, construct_loss
from src.datasets import load_dataset
from src.gan.update_g import UpdateGeneratorGAN
from src.metrics import fid
from src.utils import MetricsLogger, group_images
from src.gan.train import train_disc, train_gen, loss_terms_to_str, evaluate
from src.utils import load_z, create_checkpoint_path, seed_worker
from src.utils.checkpoint import checkpoint_gan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        required=True, help="Config file")
    parser.add_argument('--pos', dest='pos_class', default=9,
                        type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=4,
                        type=int, help='Negative class for binary classification')
    parser.add_argument('--dataset', dest='dataset',
                        default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    config = read_config(args.config_path)
    device = torch.device(config["device"])

    if args.pos_class is not None and args.neg_class is not None:
        pos_class = args.pos_class
        neg_class = args.neg_class
    else:
        print('No positive and or negative class given!')
        exit()

    dataset, num_classes, img_size = load_dataset(
        args.dataset, config["data-dir"], pos_class, neg_class)

    n_disc_iters = config['train']['step-1']['disc-iters']

    test_noise, test_noise_conf = load_z(config['test-noise'])
    batch_size = config['train']['step-1']['batch-size']

    fid_stats_path = f"{os.environ['FILESDIR']}/data/fid-stats/stats.inception.{args.dataset}.{pos_class}v{neg_class}.npz"

    fid_stats_mu, fid_stat_sigma = fid.load_statistics_from_path(fid_stats_path)
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), fid_stats_mu, fid_stat_sigma, device=device)

    fid_metrics = {
        'fid': original_fid
    }

    fixed_noise = torch.randn(
        config['fixed-noise'], config["model"]["z_dim"], device=device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=config["num-workers"], worker_init_fn=seed_worker)

    log_every_g_iter = 50

    config["project"] = f"{config['project']}-{pos_class}v{neg_class}"
    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(config, run_id)

    wandb.init(project=config["project"],
               group=config["name"],
               entity=os.environ['ENTITY'],
               job_type='step-1',
               name=f'{run_id}-step-1',
               config={
                   'id': run_id,
                   'gan': config["model"],
                   'train': config["train"]["step-1"],
               })

    train_metrics = MetricsLogger(prefix='train')
    eval_metrics = MetricsLogger(prefix='eval')

    train_metrics.add('G_loss', iteration_metric=True)
    train_metrics.add('D_loss', iteration_metric=True)

    if not os.path.exists(f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{pos_class}v{neg_class}"):
        os.makedirs(f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{pos_class}v{neg_class}")

    def random_search(param_distributions):
        best_score = float('inf')
        time_limit = 72000
        start_time = time.time()
        i = 0

        while True:
            params = {param: distribution.rvs() for param, distribution in param_distributions.items()}

            current_score, G, D, g_opt, d_opt, train_state = evaluate_model_with_params(params, i)

            if current_score < best_score:
                best_score = current_score
                config_checkpoint_dir = os.path.join(cp_dir, 'best-random-search-config')
                checkpoint_gan(
                    G, D, g_opt, d_opt, train_state,
                    {"eval": eval_metrics.stats, "train": train_metrics.stats}, config,
                    output_dir=config_checkpoint_dir)
                with open(f'{os.environ["FILESDIR"]}/step-1-best-random-search-config-{pos_class}v{neg_class}.txt',
                          'w') as file:
                    file.write(os.path.join(config_checkpoint_dir))

            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                print("Time limit reached. Stopping the random search.")
                break

            i += 1

    # Example function to evaluate model with given parameters
    def evaluate_model_with_params(params, iteration):
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

        epochs = 11
        param_scores = {}
        for epoch in range(1, epochs):
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
                              % (epoch, epochs - 1, curr_g_iter, g_iters_per_epoch, g_loss.item(),
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
            param_scores[epoch - 1] = eval_metrics.stats['fid'][epoch - 1]

        torch.save(param_scores,
                   f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{pos_class}v{neg_class}/param_scores_random_search_step1_iteration_{iteration}.pt")

        return eval_metrics.stats['fid'][epochs - 2], G, D, g_opt, d_opt, train_state

    param_distributions = {
        'g_lr': loguniform(1e-5, 1e-2),  # Logarithmic distribution for learning rates
        'd_lr': loguniform(1e-5, 1e-2),  # Logarithmic distribution for learning rates
        'g_beta1': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'd_beta1': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'g_beta2': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'd_beta2': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'n_blocks': randint(low=1, high=5),  # Discrete uniform distribution between 1 and 5
    }

    random_search(param_distributions)


if __name__ == '__main__':
    main()
