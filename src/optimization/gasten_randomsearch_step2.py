import torch
from torch.optim import Adam
import argparse
from dotenv import load_dotenv
import wandb
import math
import os
import random
from scipy.stats import uniform, randint
import json

from src.metrics import fid, LossSecondTerm
from src.gan.update_g import UpdateGeneratorGASTEN
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.classifier import ClassifierCache
from src.utils.config import read_config
from src.gan import construct_gan, construct_loss
from src.datasets import load_dataset
import math
from src.utils import MetricsLogger, group_images
from src.gan.train import train_disc, train_gen, loss_terms_to_str, evaluate
import json
from src.utils import load_z, setup_reprod, create_checkpoint_path, seed_worker
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
    parser.add_argument('--fid-stats', dest='fid_stats')
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

    classifiers = os.listdir(os.path.join(os.environ['FILESDIR'], 'models', f"{args.dataset}.{pos_class}v{neg_class}"))
    dataset, num_classes, img_size = load_dataset(
        args.dataset, config["data-dir"], pos_class, neg_class)

    config["model"]["image-size"] = img_size

    n_disc_iters = config['train']['step-2']['disc-iters']

    test_noise, test_noise_conf = load_z(config['test-noise'])
    batch_size = config['train']['step-2']['batch-size']

    mu, sigma = fid.load_statistics_from_path(args.fid_stats)
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    fixed_noise = torch.randn(
        config['fixed-noise'], config["model"]["z_dim"], device=device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=config["num-workers"], worker_init_fn=seed_worker)

    log_every_g_iter = 50

    with open(f'step-1-best-config--random-search-{pos_class}v{neg_class}.txt', 'r') as file:
        gan_path = file.read()

    if not os.path.exists(gan_path):
        print(f" WARNING: gan not found.")
        exit()

    print("Loading GAN from {} ...".format(gan_path))
    with open(os.path.join(gan_path, 'config.json'), 'r') as config_file:
        config_from_path = json.load(config_file)

    model_params = config_from_path['model']

    gen_cp = torch.load(os.path.join(
        gan_path, 'generator.pth'), map_location=device)
    dis_cp = torch.load(os.path.join(
        gan_path, 'discriminator.pth'), map_location=device)

    G, D = construct_gan(
        model_params, img_size, device=device)

    config["project"] = f"{config['project']}-{pos_class}v{neg_class}"
    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(config, run_id)

    wandb.init(project=config["project"],
               group=config["name"],
               entity=os.environ['ENTITY'],
               job_type='step-2',
               name=f'{run_id}-step-2',
               config={
                   'id': run_id,
                   'gan': config["model"],
                   'train': config["train"]["step-1"],
               })

    train_metrics = MetricsLogger(prefix='train')
    eval_metrics = MetricsLogger(prefix='eval')

    train_metrics.add('G_loss', iteration_metric=True)
    train_metrics.add('D_loss', iteration_metric=True)

    def random_search(param_distributions, num_iterations):
        best_score = float('-inf')
        best_params = None

        param_scores = {}

        for i in range(num_iterations):
            params = {param: distribution.rvs() for param, distribution in param_distributions.items()}

            # Replace this part with your model training and evaluation
            current_score, G, D, g_opt, d_opt, train_state = evaluate_model_with_params(params)

            if current_score > best_score:
                best_score = current_score
                config_checkpoint_dir = os.path.join(cp_dir, 'best-grid-search-config')
                checkpoint_gan(
                    G, D, g_opt, d_opt, train_state,
                    {"eval": eval_metrics.stats, "train": train_metrics.stats}, config,
                    output_dir=config_checkpoint_dir)
                with open(f'step-1-best-grid-search-config-{pos_class}v{neg_class}.txt', 'w') as file:
                    file.write(os.path.join(cp_dir))

            param_scores[i] = current_score

        torch.save(param_scores, f"grid_search/param_scores_gridsearch_step1.pt")
        return best_params

    # Example function to evaluate model with given parameters
    def evaluate_model_with_params(params):
        C, C_params, C_stats, C_args = construct_classifier_from_checkpoint(
            params['classifier'], device=device)
        C.to(device)
        C.eval()
        C.output_feature_maps = True

        class_cache = ClassifierCache(C)

        conf_dist = LossSecondTerm(class_cache)

        fid_metrics = {
            'fid': original_fid,
            'conf_dist': conf_dist,
        }

        g_opt = Adam(G.parameters(), lr=params['g_lr'], betas=(params['g_beta1'], params['g_beta2']))
        d_opt = Adam(D.parameters(), lr=params['d_lr'], betas=(params['d_beta1'], params['d_beta2']))

        train_state = {
            'epoch': 0,
            'best_epoch': 0,
            'best_epoch_metric': float('inf'),
        }

        G.load_state_dict(gen_cp['state'])
        D.load_state_dict(dis_cp['state'])
        g_opt.load_state_dict(gen_cp['optimizer'])
        d_opt.load_state_dict(dis_cp['optimizer'])

        G.eval()
        D.eval()

        g_crit, d_crit = construct_loss(config_from_path["model"]["loss"], D)

        g_updater = UpdateGeneratorGASTEN(g_crit, C, alpha=params['weight'])

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

        for epoch in range(1, 41):
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
                              % (epoch, 40, curr_g_iter, g_iters_per_epoch, g_loss.item(),
                                 loss_terms_to_str(g_loss_terms), d_loss.item(),
                                 loss_terms_to_str(d_loss_terms)))

            ###
            # Sample images
            ###
            with torch.no_grad():
                G.eval()
                fake = G(fixed_noise).detach().cpu()
                G.train()

            img = group_images(fake, classifier=C, device=device)
            eval_metrics.log_image('samples', img)

            ###
            # Evaluate after epoch
            ###
            train_metrics.finalize_epoch()

            evaluate(G, fid_metrics, eval_metrics, batch_size,
                     test_noise, device, None)

            eval_metrics.finalize_epoch()
        return (eval_metrics.stats['fid'][40], eval_metrics.stats['conf_dist'][40]), G, D, g_opt, d_opt, train_state

    param_distributions = {
        'g_lr': uniform(loc=0.0001, scale=0.001),  # Uniform distribution between 0.0001 and 0.001
        'd_lr': uniform(loc=0.0001, scale=0.001),  # Uniform distribution between 0.0001 and 0.001
        'g_beta1': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'd_beta1': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'g_beta2': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'd_beta2': uniform(loc=0.1, scale=0.8),  # Uniform distribution between 0.1 and 0.9
        'weights': randint(low=1, high=30),
        'classifier': classifiers
    }

    best_params = random_search(param_distributions, num_iterations=10)
    print("Best Hyperparameters:", best_params)


if __name__ == '__main__':
    main()
