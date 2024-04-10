import os
from dotenv import load_dotenv
from src.utils.config import read_config_clustering
from src.clustering.test_aux import get_gan_path, parse_args, get_clustering_path
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint
from src.metrics import fid
import wandb
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import json
import torch.optim as optim
from src.gan import construct_gan


def load_gasten(config, classifier, best_config_optim):
    """
    load information from previous step
    """
    device = config["device"]
    # get classifier name
    classifier_name = classifier.split("/")[-1]
    # get GAN
    G = None
    if 'gan_path' in config['gasten']:
        gan_path = config['gasten']['gan_path']

        with open(os.path.join(gan_path, 'config.json'), 'r') as config_file:
            gasten_config = json.load(config_file)

        model_params = gasten_config['model']
        print(model_params)

        gen_cp = torch.load(os.path.join(
            gan_path, 'generator.pth'), map_location=device)

        G, D = construct_gan(
            model_params, tuple(gasten_config['model']['image-size']), device=device)

        g_optim = optim.Adam(G.parameters(), lr=best_config_optim["g_lr"], betas=(
            best_config_optim["g_beta1"], best_config_optim["g_beta2"]))

        G.load_state_dict(gen_cp['state'])
        g_optim.load_state_dict(gen_cp['optimizer'])

        G.eval()
    else:
        gan_path = get_gan_path(config, classifier_name)
        netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)

    netG = G
    # get classifier
    C, _, _, _ = construct_classifier_from_checkpoint(classifier, device=device)
    C.eval()
    # remove last layer of classifier to get the embeddings
    C_emb = torch.nn.Sequential(*list(C.children())[0][:-1])
    C_emb.eval()

    return netG, C, C_emb, classifier_name


def save_gasten_images(config, classifier, images, classifier_name):
    """
    save embeddings and images for next step
    """
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    torch.save(classifier, f"{path}/classifier_embeddings.pt")
    thr = int(config['clustering']['acd']*10)
    torch.save(images, f"{path}/images_acd_{thr}.pt")


def generate_embeddings(config, netG, C, C_emb, classifier_name):

    device = config["device"]
    batch_size = config['batch-size']

    config_run = {
        'step': 'image_generation',
        'classifier_name': classifier_name,
        'gasten': {
            'epoch1': config['gasten']['epoch']['step-1'],
            'epoch2': config['gasten']['epoch']['step-2'],
            'weight': config['gasten']['weight']
        },
        'probabilities': {
            'min': 0.5 - config['clustering']['acd'],
            'max': 0.5 + config['clustering']['acd']
        },
        'generated_images': config['clustering']['fixed-noise']
    }

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type='step-3-amb_img_generation',
                name=f"{config['gasten']['run-id']}-{classifier_name}_{config['tag']}",
                tags=[config["tag"]],
                config=config_run)

    # prepare FID calculation
    if config['compute-fid']:
        mu, sigma = fid.load_statistics_from_path(config['dir']['fid-stats'])
        fm_fn, dims = fid.get_inception_feature_map_fn(device)
        fid_metric = fid.FID(fm_fn, dims, config_run['generated_images'], mu, sigma, device=device)

    # create fake images
    test_noise = torch.randn(config_run['generated_images'], config["clustering"]["z-dim"], device=device)
    noise_loader = DataLoader(TensorDataset(test_noise), batch_size=batch_size, shuffle=False)
    images_array = []
    for idx, batch in enumerate(tqdm(noise_loader, desc='Evaluating fake images')):
        # generate images
        with torch.no_grad():
            netG.eval()
            batch_images = netG(*batch)
        
        # calculate FID score - all images
        if config['compute-fid']:
            max_size = min(idx*batch_size, config_run['generated_images'])
            fid_metric.update(batch_images, (idx*batch_size, max_size))

        images_array.append(batch_images)

    # Concatenate batches into a single array
    images = torch.cat(images_array, dim=0)

    # FID for fake images
    if config['compute-fid'] & (images.shape[0]>=2048):
        wandb.log({"fid_score_all": fid_metric.finalize()})
        fid_metric.reset()

    # apply classifier to fake images
    with torch.no_grad():
        pred = C(images).cpu().detach().numpy()

    # filter images so that ACD < threshold
    mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
    syn_images_f = images[mask]

    # count the ambig images
    n_amb_img = syn_images_f.shape[0]
    wandb.log({"n_ambiguous_images": n_amb_img})

    # calculate FID score in batches - ambiguous images
    if config['compute-fid'] & (n_amb_img>=2048):
        image_loader = DataLoader(TensorDataset(syn_images_f), batch_size=batch_size, shuffle=False)
        for idx, batch in enumerate(tqdm(image_loader, desc='Evaluating ambiguous fake images')):
            max_size = min(idx*batch_size, config_run['generated_images'])
            fid_metric.update(*batch, (idx*batch_size, max_size))
    
        wandb.log({"fid_score_ambiguous": fid_metric.finalize()})
        fid_metric.reset()

    # get embeddings
    with torch.no_grad():
        syn_embeddings_f = C_emb(syn_images_f)

    #visualize_embeddings(config, C_emb, pred[mask], syn_embeddings_f)

    # close wandb
    wandb.finish()

    return syn_images_f, syn_embeddings_f


if __name__ == "__main__":
    # setup
    string1 = '''{
    "project": "gasten_20231211",
    "name": "mnist-7v1",
    "tag": "v4",
    "device": "cpu",
    "batch-size": 64,
    "checkpoint": "TRUE",
    "compute-fid": "TRUE",
    "dir": {
        "data": "tools/data",
        "clustering": "tools/data/clustering",
        "fid-stats": "tools/data/fid-stats/stats.inception.mnist.8v9.npz"
    },
    "dataset": {
        "name": "mnist",
        "binary": {
            "pos": "8",
            "neg": "9"
        }
    },
    "gasten": {
        "epoch": {
            "step-1": 5,
            "step-2": 10
        },
        "gan_path": "tools/out/auto_gasten/optim/Apr10T13-50_21vgy91t/1"
    },
    "clustering": {
        "z-dim": 64,
        "fixed-noise": 15000,
        "acd": 0.1,
        "n-iter": 20,
        "options": [
            {
                "dim-reduction": "umap",
                "clustering": "gmm"
            },
            {
                "dim-reduction": "umap",
                "clustering": "hdbscan"
            }
        ]
    },
    "prototypes": {
        "type": [
            "medoid"
        ]
    }
}'''
    string2 = "{'classifier': 'tools/models/mnist.8v9/cnn-4-1.1298', 'd_beta1': 0.7436704297351775, 'd_beta2': 0.6424870384644795, 'd_lr': 0.0005903948646972072, 'g_beta1': 0.4812893194050143, 'g_beta2': 0.6813047017599905, 'g_lr': 0.0004938284901364233, 'n_blocks': 5, 'weight': 29}"
    config = json.loads(string1)
    best_config = json.loads(string2.replace("'", '"'))
    load_gasten(config, best_config['classifier'], best_config)
    # load_dotenv()
    # args = parse_args()
    # config = read_config_clustering(args.config)
    # for classifier in config['gasten']['classifier']:
    #     netG, C, C_emb, classifier_name = load_gasten(config, classifier)
    #     images, _ = generate_embeddings(config, netG, C, C_emb, classifier_name)
    #     if config["checkpoint"]:
    #         save_gasten_images(config, C_emb, images, classifier_name)
