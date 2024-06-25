import torch
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical
from smac import HyperbandFacade, Scenario
from torch.optim import Adam
import argparse
from dotenv import load_dotenv
import wandb
import os
from smac.multi_objective.parego import ParEGO
from typing import Dict
import numpy as np

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


def calculate_curvature(points):
    """ Calculate the curvature of the given points """
    n_points = len(points)
    curvatures = np.zeros(n_points)

    for i in range(1, n_points - 1):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]

        v1 = p2 - p1
        v2 = p3 - p2

        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        curvatures[i] = angle

    return curvatures


def find_knee_point(data):
    sorted_data = data[np.argsort(data[:, 0])]
    curvatures = calculate_curvature(sorted_data)
    knee_index = np.argmax(curvatures)

    return sorted_data[knee_index]


def list_of_strings(arg):
    return arg.split(',')


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

    classifiers = os.listdir(os.path.join(os.environ['FILESDIR'], 'models', f"{args.dataset}.{pos_class}v{neg_class}"))
    classifier_paths = [f"{os.environ['FILESDIR']}/models/{args.dataset}.{pos_class}v{neg_class}/{classifier}"
                        for classifier in classifiers]

    dataset, num_classes, img_size = load_dataset(
        args.dataset, config["data-dir"], pos_class, neg_class)

    config["model"]["image-size"] = img_size

    n_disc_iters = config['train']['step-2']['disc-iters']

    test_noise, test_noise_conf = load_z(config['test-noise'])
    batch_size = config['train']['step-2']['batch-size']

    fid_stats_path = f"{os.environ['FILESDIR']}/data/fid-stats/stats.inception.{args.dataset}.{pos_class}v{neg_class}.npz"

    mu, sigma = fid.load_statistics_from_path(fid_stats_path)
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    fixed_noise = torch.randn(
        config['fixed-noise'], config["model"]["z_dim"], device=device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=config["num-workers"], worker_init_fn=seed_worker)

    log_every_g_iter = 50

    with open(f'{os.environ["FILESDIR"]}/step-1-best-config-hyperband-{pos_class}v{neg_class}.txt', 'r') as file:
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

    def train(params: Configuration, seed: int, budget: int) -> Dict[str, float]:
        setup_reprod(seed)

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

            img = group_images(fake, classifier=C, device=device)
            eval_metrics.log_image('samples', img)

            ###
            # Evaluate after epoch
            ###
            train_metrics.finalize_epoch()

            evaluate(G, fid_metrics, eval_metrics, batch_size,
                     test_noise, device, None)

            eval_metrics.finalize_epoch()

        config_checkpoint_dir = os.path.join(cp_dir, str(params.config_id))
        checkpoint_gan(
            G, D, g_opt, d_opt, train_state,
            {"eval": eval_metrics.stats, "train": train_metrics.stats}, config, output_dir=config_checkpoint_dir)

        return {
            "fid": eval_metrics.stats['fid'][round(budget) - 1],
            "confusion_distance": eval_metrics.stats['conf_dist'][round(budget) - 1],
        }

    G_lr = Float("g_lr", (1e-4, 1e-3), default=0.0002)
    D_lr = Float("d_lr", (1e-4, 1e-3), default=0.0002)
    G_beta1 = Float("g_beta1", (0.1, 1), default=0.5)
    D_beta1 = Float("d_beta1", (0.1, 1), default=0.5)
    G_beta2 = Float("g_beta2", (0.1, 1), default=0.999)
    D_beta2 = Float("d_beta2", (0.1, 1), default=0.999)
    weights = Integer("weight", (1, 30), default=25)
    classifiers = Categorical('classifier', classifier_paths)

    configspace = ConfigurationSpace()
    configspace.add_hyperparameters([G_lr, D_lr, G_beta1, D_beta1, G_beta2, D_beta2, weights, classifiers])

    objectives = ["fid", "confusion_distance"]

    scenario = Scenario(configspace, objectives=objectives, deterministic=True, min_budget=2,
                        max_budget=40, walltime_limit=288000, n_trials=10000)
    multi_objective_algorithm = ParEGO(scenario)
    smac = HyperbandFacade(scenario, train, multi_objective_algorithm=multi_objective_algorithm, overwrite=True)
    incumbents = smac.optimize()

    data = np.array(smac.intensifier.trajectory[len(smac.intensifier.trajectory) - 1].costs)

    knee_point = find_knee_point(data)

    print("Configs from the Pareto front (incumbents):")
    knee_point_config = 0
    for i in range(len(smac.intensifier.trajectory[len(smac.intensifier.trajectory) - 1].costs)):
        print("Best Configuration:", incumbents[i].get_dictionary())
        print("Configuration id:", incumbents[i].config_id)
        print("Cost:", smac.intensifier.trajectory[len(smac.intensifier.trajectory) - 1].costs[i])
        if smac.intensifier.trajectory[len(smac.intensifier.trajectory) - 1].costs[i] == list(knee_point):
            knee_point_config = i

    print(f"Knee point: {knee_point}")

    config_with_lowest_conf_dist = smac.intensifier.trajectory[len(smac.intensifier.trajectory) - 1].config_ids[
        knee_point_config]
    with open(f'{os.environ["FILESDIR"]}/step-2-hyperband-best-config-{args.dataset}-{pos_class}v{neg_class}.txt', 'w') as file:
        file.write(os.path.join(cp_dir, str(config_with_lowest_conf_dist)) + '\n')
        for incumbent in incumbents:
            if config_with_lowest_conf_dist == incumbent.config_id:
                file.write(str(incumbent.get_dictionary()))

    wandb.finish()


if __name__ == '__main__':
    main()
