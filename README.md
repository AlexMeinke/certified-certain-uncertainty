# Towards neural networks that provably know when they don't know

This repository contains the code that was used to obtain the results reported in https://arxiv.org/abs/1909.12180. In it we propose a *Certified Certain Uncertainty* (CCU) model with which one can train deep neural networks that provably make low-confidence predictions far away from the training data.

## Training the models

Before training a CCU model, one has to first initialize a Gaussian mixture model on the datasets from the in- and out-distribution [80 Million Tiny Images](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin). 
```
python gen_gmm.py --dataset MNIST --PCA 1 --augm_flag 1
```
The PCA option refers to the fact that we use a modified distance metric. 

Most models in the paper are trained via the script in **run_training.py**. Hyper parameters can be passed as options, but defaults are stored in **model_params.py**. For example the following lines train a plain model, [ACET](https://arxiv.org/abs/1812.05720) model and a [CCU](https://arxiv.org/abs/1909.12180) model on augmented data on MNIST.

```
python run_training.py --dataset MNIST --train_type plain --augm_flag 1
python run_training.py --dataset MNIST --train_type ACET --augm_flag 1
python run_training.py --dataset MNIST --train_type CEDA_GMM --augm_flag 1 --use_gmm 1 --PCA 1 --fit_out 1
```
For all commands (except **gen_gmm.py**) one can specify, which GPU to train on via the --gpu option. All model paths for models that one wishes to use as base models or for testing should be stored in **model_paths.py**.

An [ODIN](https://arxiv.org/abs/1706.02690) model and single layer [Maha](https://arxiv.org/abs/1807.03888) model can be generated from the pretrained base model by running their respective notebooks.

The [GAN](https://arxiv.org/abs/1711.09325) model is trained using the code in https://github.com/alinlab/Confident_classifier and then only imported using the GAN notebook. The same goes for [OE](https://arxiv.org/abs/1812.04606) which we train with https://github.com/hendrycks/outlier-exposure except that we substitute in our architecture.

An [MCD](https://arxiv.org/abs/1506.02142) model can be trained via 
```
python train_mcdo.py --dataset MNIST
```

## Testing the models

The out-of-distribution detection statistics (Table 2) are generated by specifying the models one wishes to test in **model_paths.py** and then running
```
python gen_eval.py --dataset MNIST
```

The result of our adversarial noise attack (Table 1) comes from
```
python gen_attack_stats.py --datasets MNIST --wide_format 1 --fit_out 1
```
where mutliple datasets could be specified. The **gen_attack_stats.py** script dumps full information, like all confidences and perturbed samples in results/backup. This path has to be specified when reproducing Figure 2 and 3 with their respective notebooks.
