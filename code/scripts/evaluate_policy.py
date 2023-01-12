import sys
sys.path.append("/iliad/u/manasis/language-diffuser/code")

import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_default_model_and_env, get_env_state_for_initial_condition, join_vis_lang
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env
from omegaconf import DictConfig, ListConfig, OmegaConf
import diffuser.utils as utils
from config.locomotion_config import Config

from diffuser.utils.arrays import to_torch, to_np, to_device

import play_lmp as models_m

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
    print(f"logging to {log_dir}")
    return log_dir


class CustomModel:
    def __init__(self, cfg):
        state_dict = torch.load(f'/iliad/u/manasis/language-diffuser/code/logs/checkpoint/state.pt',
                                map_location=Config.device)

        dataset_config = utils.Config(
            Config.loader,
            savepath='dataset_config.pkl',
            cfg=cfg,
            train_flag=False,
            horizon=Config.horizon,
            normalizer=Config.normalizer,
            #preprocess_fns=Config.preprocess_fns,
            use_padding=Config.use_padding,
            max_path_length=Config.max_path_length,
            include_returns=Config.include_returns,
            #returns_scale=Config.returns_scale,
            #discount=Config.discount,
            #termination_penalty=Config.termination_penalty,
        )
        dataset = dataset_config()
        self.dataset = dataset

        observation_dim = dataset.observation_dim
        self.observation_dim = observation_dim
        action_dim = dataset.action_dim

        if Config.diffusion == 'models.GaussianInvDynDiffusion':
            transition_dim = observation_dim
        else:
            transition_dim = observation_dim + action_dim

        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=transition_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            dim=Config.dim,
            returns_condition=Config.returns_condition,
            device=Config.device,
        )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            hidden_dim=Config.hidden_dim,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            device=Config.device,
            condition_guidance_w=Config.condition_guidance_w,
        )

        trainer_config = utils.Config(
            utils.Trainer,
            savepath='trainer_config.pkl',
            train_batch_size=Config.batch_size,
            train_lr=Config.learning_rate,
            gradient_accumulate_every=Config.gradient_accumulate_every,
            ema_decay=Config.ema_decay,
            sample_freq=Config.sample_freq,
            save_freq=Config.save_freq,
            log_freq=Config.log_freq,
            label_freq=int(Config.n_train_steps // Config.n_saves),
            save_parallel=Config.save_parallel,
            bucket=Config.bucket,
            n_reference=Config.n_reference,
            train_device=Config.device,
        )
        """render_config = utils.Config(
            Config.renderer,
            savepath='render_config.pkl',
            env=Config.dataset,
        )"""
        model = model_config()
        diffusion = diffusion_config(model)
        #renderer = render_config()
        trainer = trainer_config(diffusion, dataset, None)
        trainer.step = state_dict['step']
        trainer.model.load_state_dict(state_dict['model'])
        trainer.ema_model.load_state_dict(state_dict['ema'])
        self.trainer = trainer

        # Encoding model load
        chk = Path("/iliad/u/manasis/conditional-diffuser/D_D_static_rgb_baseline/mcil_baseline.ckpt") #get_last_checkpoint(Path.cwd())
        #import pdb;pdb.set_trace()

        # Load Model
        if chk is not None:
            encoding_model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(chk.as_posix())
        else:
            encoding_model = hydra.utils.instantiate(cfg.model)
        self.encoding_model = encoding_model

    def reset(self):
        pass

    def step(self, obs, goal):
        #import pdb;pdb.set_trace()
        rgb_obs = torch.Tensor(np.expand_dims(obs['rgb_obs']['rgb_static'].transpose(2, 0, 1), (0, 1)))
        rgb_obs_dict = {'rgb_static': rgb_obs}
        robot_obs = np.concatenate((obs["robot_obs"][:7], obs["robot_obs"][14:15]))
        robot_obs = torch.Tensor(robot_obs.reshape(1, 1, len(robot_obs)))
        perceptual_emb = self.encoding_model.perceptual_encoder(rgb_obs_dict, {}, robot_obs).squeeze(0).detach().numpy()

        #perceptual_emb = self.encoding_model.perceptual_encoder(obs['rgb_obs'], obs["depth_obs"], obs["robot_obs"]).squeeze().detach().numpy() #torch.Size([32, 32, 3, 200, 200]) --> torch.Size([32, 32, 72])
        obs = self.dataset.normalizer.normalize(perceptual_emb, 'observations')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        conditions = {0: to_torch(obs, device=device)}
        samples = self.trainer.ema_model.conditional_sample(conditions, returns=None) #goal)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*self.observation_dim)
        action = self.trainer.ema_model.inv_model(obs_comb)
        #action = action.reshape(len(action[0]), 1)
        action = action.squeeze()

        samples = to_np(samples)
        action = to_np(action)

        action = self.dataset.normalizer.unnormalize(action, 'actions')
        return action


class CustomLangEmbeddings:
    def __init__(self):
        logger.warning("Please implement these methods in order to use your own language embeddings")
        raise NotImplementedError

    def get_lang_goal(self, task_annotation):
        """
        Args:
             task_annotation: langauge annotation
        Returns:

        """
        raise NotImplementedError


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, args):
    log_dir = get_log_dir(args.log_dir)

    sequences = get_sequences(args.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem.split("=")[1]
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")

    for checkpoint, plan_dict in plan_dicts.items():
        epoch = checkpoint.stem.split("=")[1]

        ids, labels, plans, latent_goals = zip(
            *[
                (i, label, latent_goal, plan)
                for i, (label, plan_list) in enumerate(plan_dict.items())
                for latent_goal, plan in plan_list
            ]
        )
        latent_goals = torch.cat(latent_goals)
        plans = torch.cat(plans)
        np.savez(
            f"{log_dir / f'tsne_data_{epoch}.npz'}", ids=ids, labels=labels, plans=plans, latent_goals=latent_goals
        )


def evaluate_policy(model, env, lang_embeddings, args):
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load("/iliad/u/manasis/language-diffuser/code/scripts/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load("/iliad/u/manasis/language-diffuser/code/scripts/conf/annotations/new_playtable_validation.yaml")

    eval_sequences = get_sequences(args.num_sequences)

    results = []
    plans = defaultdict(list)

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, args, plans
        )
        results.append(result)
        if not args.debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    return results, plans


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, args, plans
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if args.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, args, subtask, lang_embeddings, val_annotations, plans)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, args, subtask, lang_embeddings, val_annotations, plans):
    if args.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    if lang_embeddings:
        goal = lang_embeddings.get_lang_goal(lang_annotation)
    else:
        goal = None
    model.reset()
    start_info = env.get_info()

    #plan, latent_goal = model.get_pp_plan_lang(obs, goal)
    #plans[subtask].append((plan.cpu(), latent_goal.cpu()))

    for step in range(args.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        if args.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if args.debug:
                print(colored("success", "green"), end=" ")
            return True
    if args.debug:
        print(colored("fail", "red"), end=" ")
    return False

class Args:
    def __init__(self):
        self.dataset_path = '/iliad/u/manasis/language-diffuser/code/calvin_debug_dataset'
        self.train_folder = None
        self.checkpoints = None
        self.checkpoint = None
        self.last_k_checkpoints =  None
        self.custom_model = True
        self.custom_lang_embeddings = False
        self.debug = False
        self.log_dir = None
        self.device = 0

def wrap_main(config_name):
    @hydra.main(config_path="conf", config_name=f"{config_name}.yaml")
    def main(cfg: DictConfig) -> None:
        seed_everything(0, workers=True)  # type:ignore

        #parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")        
        #args = parser.parse_args()
        args = Args()
        args.dataset_path = '/iliad/u/manasis/language-diffuser/code/calvin_debug_dataset'
        args.train_folder = None
        args.checkpoints = None
        args.checkpoint = None
        args.last_k_checkpoints =  None
        args.custom_model = True
        args.custom_lang_embeddings = False
        args.debug = False
        args.log_dir = None
        args.device = 0 

        """parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

        # arguments for loading default model
        parser.add_argument(
            "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
        )
        parser.add_argument(
            "--checkpoints",
            type=str,
            default=None,
            help="Comma separated list of epochs for which checkpoints will be loaded",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path of the checkpoint",
        )
        parser.add_argument(
            "--last_k_checkpoints",
            type=int,
            help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
        )

        # arguments for loading custom model or custom language embeddings
        parser.add_argument(
            "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
        )
        parser.add_argument("--custom_lang_embeddings", action="store_true", help="Use custom language embeddings.")

        parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

        parser.add_argument("--log_dir", default=None, type=str, help="Where to log the evaluation results.")

        parser.add_argument("--device", default=0, type=int, help="CUDA device")"""

        # Do not change
        args.ep_len = 360
        args.num_sequences = 1000

        lang_embeddings = None
        if args.custom_lang_embeddings:
            lang_embeddings = CustomLangEmbeddings()

        # evaluate a custom model
        if args.custom_model:
            model = CustomModel(cfg)
            env = make_env(args.dataset_path)
            evaluate_policy(model, env, lang_embeddings, args)
        else:
            assert "train_folder" in args

            checkpoints = []
            if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
                print("Evaluating model with last checkpoint.")
                checkpoints = [get_last_checkpoint(Path(args.train_folder))]
            elif args.checkpoints is not None:
                print(f"Evaluating model with checkpoints {args.checkpoints}.")
                checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
            elif args.checkpoints is None and args.last_k_checkpoints is not None:
                print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
                checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
            elif args.checkpoint is not None:
                checkpoints = [Path(args.checkpoint)]

            env = None
            results = {}
            plans = {}
            for checkpoint in checkpoints:
                model, env, _, lang_embeddings = get_default_model_and_env(
                    args.train_folder,
                    args.dataset_path,
                    checkpoint,
                    env=env,
                    lang_embeddings=lang_embeddings,
                    device_id=args.device,
                )
                results[checkpoint], plans[checkpoint] = evaluate_policy(model, env, lang_embeddings, args)

            print_and_save(results, plans, args)
    main()

def setup_config():
    config_str = next((x for x in sys.argv if "config_name" in x), None)
    if config_str is not None:
        config_name = config_str.split("=")[1]
        sys.argv.remove(config_str)
        os.environ["HYDRA_CONFIG_NAME"] = config_name
        return config_name
    elif "HYDRA_CONFIG_NAME" in os.environ:
        return os.environ["HYDRA_CONFIG_NAME"]
    else:
        return "config"


if __name__ == "__main__":
    conf = setup_config()
    wrap_main(conf)