from collections import namedtuple
import numpy as np
import torch
import pdb

#from .preprocessing import get_preprocess_fn
#from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

from pathlib import Path
import play_lmp as models_m
import hydra


RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, #env='hopper-medium-replay',
        cfg=None,
        train_flag=True,
        horizon=64,
        normalizer='LimitsNormalizer', #preprocess_fns=[], 
        max_path_length=1000,
        #max_n_episodes=10000, #termination_penalty=0, 
        max_n_episodes=1013111, #termination_penalty=0, 
        use_padding=True, #discount=0.99, returns_scale=1000, 
        include_returns=False):

        print("\n\n\n\n\n\n\nYOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n\n\n\n\n\n")

        #self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        #self.env = env = load_environment(env)
        #self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        #self.discount = discount
        #self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        
        """Dataset initialization"""
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.prepare_data()
        datamodule.setup()
        if train_flag:
            calvin_dataloader = datamodule.train_dataloader()['lang']
        else:
            calvin_dataloader = datamodule.val_dataloader()

        """Model initialization"""
        chk = Path("/iliad/u/manasis/conditional-diffuser/D_D_static_rgb_baseline/mcil_baseline.ckpt") #get_last_checkpoint(Path.cwd())
        #import pdb;pdb.set_trace()

        # Load Model
        if chk is not None:
            model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(chk.as_posix())
        else:
            model = hydra.utils.instantiate(cfg.model)
                
        #itr = sequence_dataset(env, self.preprocess_fn)

        # self.max_path_length
        # max_n_episodes = 10000
        # env and preprocess_fn are not needed
        # in episode, 'next_observation' is not needed as no preprocess_fn is being applied
        # timeouts and terminals were only applied to returns, not needed
        # infos/qpos, qvel, etc. are used for rendering (images), comment that out of training

        """for i, episode in enumerate(itr):
            import pdb;pdb.set_trace()
            # here is where to do all the changes
            fields.add_path(episode)"""

        """Creating embeddings initialization"""
        fields = ReplayBuffer(max_n_episodes, max_path_length) #, termination_penalty)
        for i, batch in enumerate(calvin_dataloader):
            episode = {}
            if train_flag:
                batch_obj = batch
            else:
                batch_obj = batch['lang']
            #import pdb;pdb.set_trace()
            
            #perceptual_emb = model.perceptual_encoder(batch_obj['rgb_obs'], batch_obj["depth_obs"], batch_obj["robot_obs"]).squeeze().detach().numpy() #torch.Size([32, 32, 3, 200, 200]) --> torch.Size([1, 32, 72])
            import pdb;pdb.set_trace()
            perceptual_emb = model.perceptual_encoder.proprio_encoder(batch_obj["robot_obs"]).squeeze(0) # torch.Size([1, 32, 32]) --> torch.Size([32, 32])
            #import pdb;pdb.set_trace()
            # try both robot_obs (15,) & partial (8,)

            latent_goal = model.language_goal(batch_obj['lang']).detach().numpy() #torch.Size([32, 384]) --> torch.Size([32, 32])
            len_hor = len(perceptual_emb)
            #latent_goal = np.tile(latent_goal, (len_hor, 1))
            action_emb = batch_obj['actions'].squeeze().detach().numpy()
            episode['observations'] = perceptual_emb
            episode['actions'] = action_emb
            episode['language'] = latent_goal
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        #import pdb;pdb.set_trace()
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            """rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)"""
            fields_language = self.fields.language[:, start:end, :]
            fields_language = np.unique(fields_language, axis=1)
            if fields_language.shape[1] != 1:
                print("\n\nError!\n\n")
                import pdb;pdb.set_trace()
            returns = fields_language[path_ind, 0]
            #import pdb;pdb.set_trace()
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)
        return batch

class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        t_step = np.random.randint(0, self.horizon)

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        traj_dim = self.action_dim + self.observation_dim

        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1

        if t_step < self.horizon-1:
            observations[t_step+1:] = 0

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
