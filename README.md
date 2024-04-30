# GASTeN Project

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Variation of GANs that, given a model, generates realistic data that is classiﬁed with low conﬁdence by a given classiﬁer. Results show that the approach is able to generate images that are closer to the frontier when compared to the original ones, but still realistic. Manual inspection conﬁrms that some of those images are confusing even for humans.

Paper: [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

## Create Virtual Environment and directories

```ssh
mamba create -n gasten python=3.10

mamba activate gasten

mamba install pip-tools

pip3 install -r requirements.txt

mkdir <file-directory>/data/clustering

mkdir <file-directory>/data/fid-stats

mkdir <file-directory>/out
```

## Run

### env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file-directory>
ENTITY=<wandb entity to track experiments>
```

### GASTeN

Run AutoGASTeN to create images in the bounday between **8** and **9**.
 
`python3 -m src.optimization --dataset mnist --pos 8 --neg 9 --config experiments/optimization/auto_gasten.yml --config_clustering experiments/clustering/mnist_7v1.yml`