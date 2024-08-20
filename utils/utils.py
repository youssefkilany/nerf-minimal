def seed_torch(seed=0):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed=0):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
