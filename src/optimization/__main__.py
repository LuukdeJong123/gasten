import itertools
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from dotenv import load_dotenv
from src.utils import create_and_store_z, gen_seed, set_seed

load_dotenv()
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', dest='dataroot',
                    default=f"{os.environ['FILESDIR']}/data", help='Dir with dataset')
parser.add_argument('--out-dir', dest='out_dir',
                    default=f"{os.environ['FILESDIR']}/models", help='Path to generated files')
parser.add_argument('--dataset', dest='dataset',
                    default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
parser.add_argument('--n-classes', dest='n_classes',
                    default=10, help='Number of classes in dataset')
parser.add_argument('--device', type=str, default='cpu',
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
parser.add_argument("--config", dest="config_path", required=True, help="Config file")
parser.add_argument("--seed", type=int, default=None)

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
        pos_class = args.pos_class
        neg_class = args.neg_class
    else:
        print('No positive and or negative class given!')
        exit()

    print(f"\nGenerating FID score for {pos_class}v{neg_class} ...")
    subprocess.run(['python', '-m', 'src.metrics.fid',
                           '--data', args.dataroot,
                           '--dataset', args.dataset,
                           '--device', args.device,
                           '--pos', str(pos_class), '--neg', str(neg_class)])

    print(f"\nGenerating classifiers for {pos_class}v{neg_class} ...")
    for clf_type, nf, epochs in itertools.product(l_clf_type, l_nf, l_epochs):
        print("\n", clf_type, nf, epochs)
        proc = subprocess.run(["python", "-m", "src.classifier.train",
                               "--device", args.device,
                               "--data-dir", args.dataroot,
                               "--out-dir", args.out_dir,
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


    test_noise, test_noise_path = create_and_store_z(
        args.out_dir, args.nz, args.z_dim,
        config={'seed': seed, 'n_z': args.nz, 'z_dim': args.z_dim})

    print("Generated test noise, stored in", test_noise_path)

    print("Start hyperparameter optimization step-1")
    subprocess.run(['python', '-m', 'src.optimization.gasten_multifidelity_optimization_step1',
                               '--config', args.config])


    print("Start hyperparameter optimization step-2")
    subprocess.run(['python', '-m', 'src.optimization.gasten_multifidelity_optimization_step2',
                               '--config', args.config])

    print("Start clustering")
    #https://github.com/inesgomes/gasten/blob/interpretability/src/clustering/__main__.py
if __name__ == '__main__':
    main()
