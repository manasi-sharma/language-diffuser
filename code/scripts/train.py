import sys
sys.path.append("/iliad/u/manasis/language-diffuser/code")
 
import diffuser.utils as utils
import torch

import hydra
import os
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer


def main(config_name, **deps):
    #def train(cfg: DictConfig) -> None:
    @hydra.main(config_path="conf", config_name=f"{config_name}.yaml")
    def train(cfg: DictConfig) -> None:

        #from ml_logger import logger, RUN
        #sys.path.append("..")
        from config.locomotion_config import Config
        #from locomotion_config import Config

        #RUN._update(deps)
        Config._update(deps)

        # logger.remove('*.pkl')
        # logger.remove("traceback.err")
        #logger.log_params(Config=vars(Config), RUN=vars(RUN))
        # logger.log_text("""
        #                 charts:
        #                 - yKey: loss
        #                   xKey: steps
        #                 - yKey: a0_loss
        #                   xKey: steps
        #                 """, filename=".charts.yml", dedent=True, overwrite=True)

        torch.backends.cudnn.benchmark = True
        #utils.set_seed(Config.seed)
        # -----------------------------------------------------------------------------#
        # ---------------------------------- dataset ----------------------------------#
        # -----------------------------------------------------------------------------#

        seed_everything(cfg.seed, workers=True)  # type: ignore

        """# Dataset
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.prepare_data()
        datamodule.setup()
        calvin_dataset = datamodule.train_datasets
        #dataloader = datamodule.val_dataloader()"""

        dataset_config = utils.Config(
            Config.loader,
            savepath='dataset_config.pkl',
            cfg=cfg,
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

        render_config = utils.Config(
            Config.renderer,
            savepath='render_config.pkl',
            env=Config.dataset,
        )

        dataset = dataset_config()
        renderer = render_config()
        observation_dim = dataset.observation_dim
        action_dim = dataset.action_dim

        # -----------------------------------------------------------------------------#
        # ------------------------------ model & trainer ------------------------------#
        # -----------------------------------------------------------------------------#
        if Config.diffusion == 'models.GaussianInvDynDiffusion':
            model_config = utils.Config(
                Config.model,
                savepath='model_config.pkl',
                horizon=Config.horizon,
                transition_dim=observation_dim,
                cond_dim=observation_dim,
                dim_mults=Config.dim_mults,
                returns_condition=Config.returns_condition,
                dim=Config.dim,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
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
                ar_inv=Config.ar_inv,
                train_only_inv=Config.train_only_inv,
                ## loss weighting
                action_weight=Config.action_weight,
                loss_weights=Config.loss_weights,
                loss_discount=Config.loss_discount,
                returns_condition=Config.returns_condition,
                condition_guidance_w=Config.condition_guidance_w,
                device=Config.device,
            )
        else:
            model_config = utils.Config(
                Config.model,
                savepath='model_config.pkl',
                horizon=Config.horizon,
                transition_dim=observation_dim + action_dim,
                cond_dim=observation_dim,
                dim_mults=Config.dim_mults,
                returns_condition=Config.returns_condition,
                dim=Config.dim,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
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
                ## loss weighting
                action_weight=Config.action_weight,
                loss_weights=Config.loss_weights,
                loss_discount=Config.loss_discount,
                returns_condition=Config.returns_condition,
                condition_guidance_w=Config.condition_guidance_w,
                device=Config.device,
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
            save_checkpoints=Config.save_checkpoints,
        )

        # -----------------------------------------------------------------------------#
        # -------------------------------- instantiate --------------------------------#
        # -----------------------------------------------------------------------------#

        model = model_config()

        diffusion = diffusion_config(model)

        trainer = trainer_config(diffusion, dataset, renderer)

        # -----------------------------------------------------------------------------#
        # ------------------------ test forward & backward pass -----------------------#
        # -----------------------------------------------------------------------------#

        utils.report_parameters(model)

        #logger.print('Testing forward...', end=' ', flush=True)
        print('Testing forward...', end=' ', flush=True)
        batch = utils.batchify(dataset[0], Config.device)
        loss, _ = diffusion.loss(*batch)
        loss.backward()
        #logger.print('✓')
        print('✓')

        # -----------------------------------------------------------------------------#
        # --------------------------------- main loop ---------------------------------#
        # -----------------------------------------------------------------------------#

        n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

        for i in range(n_epochs):
            #logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
            print(f'Epoch {i} / {n_epochs}')
            trainer.train(n_train_steps=Config.n_steps_per_epoch)

    train()

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
    main(conf)
