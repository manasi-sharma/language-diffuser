import torch

#from params_proto.neo_proto import ParamsProto, PrefixProto, Proto
from params_proto.proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = '/iliad/u/manasis/language-diffuser/code/logs/'
    dataset = 'hopper-medium-expert-v2'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianInvDynDiffusion'
    horizon = 32 #CHANGED
    n_diffusion_steps = 1000
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True #CHANGED
    discount = 0.99
    max_path_length = 33
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 1 #10000
    loss_type = 'l2'
    n_train_steps = 5 #1e6
    batch_size = 8
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 10000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False

    def __init__(self):
        super().__init__()
        pass
