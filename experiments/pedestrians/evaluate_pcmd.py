import sys
import os
import dill
import json
import argparse
import torch
import numpy as np

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import pdb

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--node_type", help="node type to evalu ate", type=str)
parser.add_argument("--seed", help="random seed", type=int,default=0)
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        ############### PCMD ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        ade_ls = []
        fde_ls = []
        print("-- Evaluating PCMD")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False,
                                               all_z_sep=True,
                                               pcmd=True)

                if not predictions:
                    continue

                batch_error_dict = evaluation.batch_pcmd(predictions,
                                                         scene.dt,
                                                         max_hl=max_hl,
                                                         ph=ph,
                                                         node_type_enum=env.NodeType,
                                                         kde=False,
                                                         map=None,
                                                         best_of=False,
                                                         prune_ph_to_future=True)
                ade_ls.append(batch_error_dict[args.node_type]['ade'])
                fde_ls.append(batch_error_dict[args.node_type]['fde'])

        ade_ls = np.vstack(ade_ls)
        fde_ls = np.vstack(fde_ls)
        ade_pcmd = np.zeros(ade_ls.shape[1])
        fde_pcmd = np.zeros(ade_ls.shape[1])
        for i in range(ade_ls.shape[1]):
            ade_pcmd[i] = ade_ls[:, :i+1].min(axis=1).mean()
            fde_pcmd[i] = fde_ls[:, :i+1].min(axis=1).mean()

        print('ade PCMD 1/M:', ade_pcmd[0], '5/M:', ade_pcmd[4], '20/M:', ade_pcmd[19])
        print('fde PCMD 1/M:', fde_pcmd[0], '5/M:', fde_pcmd[4], '20/M:', fde_pcmd[19])
