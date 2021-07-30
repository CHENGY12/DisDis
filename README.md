# Personalized Trajectory Prediction via Distribution Discrimination (DisDis)
The official PyTorch code implementation of "Personalized Trajectory Prediction via Distribution Discrimination" in ICCV 2021,[arxiv](https://arxiv.org/pdf/2107.14204.pdf).

## Introduction
The motivation of DisDis is to learn the latent distribution to represent different motion patterns, where the motion pattern of each person is personalized due to his/her habit. We learn the distribution discriminator in a self-supervised manner, which encourages the latent variable distributions of the same motion pattern to be similar while pushing the ones of the different motion patterns away. DisDis is a plug-and-play module which could be integrated with existing multi-modal stochastic predictive models to enhance the discriminative ability of latent distribution. Besides, we propose a new evaluation metric for stochastic trajectory prediction methods. We calculate the probability cumulative minimum distance (PCMD) curve to comprehensively and stably evaluate the learned model and latent distribution, which cumulatively selects the minimum distance between sampled trajectories and ground-truth trajectories from high probability to low probability. PCMD considers the predictions with corresponding probabilities and evaluates the prediction model under the whole latent distribution.

![image](https://github.com/CHENGY12/DisDis/blob/main/images/model_DisDis.png)
Figure 1. Training process for the DisDis method. DisDis regards the latent distribution as the motion pattern and optimizes the trajectories with the same motion pattern to be close while the ones with different patterns are pushed away, where the same latent distributions are in the same color. For a given history trajectory, DisDis predicts a latent distribution as the motion pattern, and takes the latent distribution as the discrimination to jointly optimize the embeddings of trajectories and latent distributions.


## Requirements
- Python 3.6+
- PyTorch 1.4

To build all the dependency, you can follow the instruction below.
```
pip install -r requirements.txt
```
Our code is based on [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus). Please cite it if it's useful.

## Dataset
The preprocessed data splits for the ETH and UCY datasets are in `experiments/pedestrians/raw/`. Before training and evaluation, execute the following to process the data. This will generate .pkl files in experiments/processed.
```
cd experiments/pedestrians
python process_data.py
```
The `train/validation/test/` splits are the same as those found in [Social GAN]( https://github.com/agrimgupta92/sgan).

## Model training

You can train the model for zara1 dataset as
```
python train.py --eval_every 10 --vis_every 1 --train_data_dict zara1_train.pkl --eval_data_dict zara1_val.pkl --offline_scene_graph yes --preprocess_workers 2 --log_dir ../experiments/pedestrians/models --log_tag _zara1_disdis --train_epochs 100 --augment --conf ../experiments/pedestrians/models/config/config_zara1.json --device cuda:0
```
The pre-trained models can be found in `experiments/pedestrians/models/`. And the model configuration is in `experiments/pedestrians/models/config/`.

## Model evaluation

To reproduce the PCMD results in Table 1, you can use
```
python evaluate_pcmd.py --node_type PEDESTRIAN --data ../processed/zara1_test.pkl --model models/zara1_pretrain --checkpoint 100
```

To use the most-likely strategy, you can use
```
python evaluate_mostlikely_z.py --node_type PEDESTRIAN --data ../processed/zara1_test.pkl --model models/zara1_pretrain --checkpoint 100
```

Welcome to use our PCMD evaluation metric in your experiments. It is a more comprehensive and stable evaluation metric for stochastic trajectory prediction methods.

## Citation

The bibtex of our paper 'Personalized Trajectory Prediction via Distribution Discrimination' is provided below:

```
@inproceedings{Disdis,
  title={Personalized Trajectory Prediction via Distribution Discrimination},
  author={Chen, Guangyi and Li, Junlong and Zhou, Nuoxing and Ren, Liangzheng and Lu, Jiwen},
  booktitle={ICCV},
  year={2021}
}
```