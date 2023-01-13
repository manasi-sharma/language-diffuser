import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import sys
sys.path.append("/iliad/u/manasis/language-diffuser/code")
import os
from pathlib import Path
import play_lmp as models_m
import torch
import diffuser.utils as utils
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device

def wrap_main(config_name):
    @hydra.main(config_path="conf", config_name=f"{config_name}.yaml")
    def main(cfg: DictConfig) -> None:

        """Loading in val dataset"""
        train_flag = False
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.prepare_data()
        datamodule.setup()
        if train_flag:
            calvin_dataloader = datamodule.train_dataloader()['lang']
        else:
            calvin_dataloader = datamodule.val_dataloader()

        """Loading in pre-trained language and vision embedding models"""
        chk = Path("/iliad/u/manasis/conditional-diffuser/D_D_static_rgb_baseline/mcil_baseline.ckpt") #get_last_checkpoint(Path.cwd())
        # Load Model
        if chk is not None:
            encoding_model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(chk.as_posix())
        else:
            encoding_model = hydra.utils.instantiate(cfg.model)

        """Loading in trained diffusion model"""
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

        observation_dim = dataset.observation_dim
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
        model = model_config()
        diffusion = diffusion_config(model)
        #renderer = render_config()
        trainer = trainer_config(diffusion, dataset, None)
        trainer.step = state_dict['step']
        trainer.model.load_state_dict(state_dict['model'])
        trainer.ema_model.load_state_dict(state_dict['ema'])


        """Inference"""
        for i, batch in enumerate(calvin_dataloader):
            episode = {}
            if train_flag:
                batch_obj = batch
            else:
                batch_obj = batch['lang']
            
            import pdb;pdb.set_trace()
            perceptual_emb = encoding_model.perceptual_encoder(batch_obj['rgb_obs'], batch_obj["depth_obs"], batch_obj["robot_obs"]).squeeze().detach().numpy() #torch.Size([32, 32, 3, 200, 200]) --> torch.Size([32, 32, 72])
            obs = dataset.normalizer.normalize(perceptual_emb, 'observations')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            conditions = {0: to_torch(obs, device=device)}
            samples = trainer.ema_model.conditional_sample(conditions, returns=None) #goal)
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*observation_dim)
            action = trainer.ema_model.inv_model(obs_comb)
            #action = action.reshape(len(action[0]), 1)
            action = action.squeeze()

            samples = to_np(samples)
            action = to_np(action)

            action = dataset.normalizer.unnormalize(action, 'actions')
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