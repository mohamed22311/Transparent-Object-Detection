from collections import namedtuple
import math
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init
import math
from functools import partial
from typing import Callable
from torch import optim

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA(object):
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models

    Keeps a moving average of everything in the model state_dict (parameters and buffers).

    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        """
        Creates an EMA model.

        Args:
            model (torch.nn.Module): The model to track with EMA.
            decay (float, optional): EMA decay rate. Defaults to 0.9999.
            tau (int, optional): EMA update ramp-up timescale. Defaults to 2000.
            updates (int, optional): Number of EMA updates performed so far. Defaults to 0.
        """

        # Create EMA model with parameters in FP32
        self.ema = deepcopy(de_parallel(model)).eval()
        self.ema.float()  # Ensure EMA parameters are in FP32

        self.updates = updates
        self.decay = namedtuple('Decay', ['fn'])(lambda x: decay * (1 - math.exp(-x / tau)))  # Decay function with ramp-up

        # Freeze EMA parameters
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model: torch.nn.Module):
        """
        Updates the EMA model with the current model state.

        Args:
            model (torch.nn.Module): The model to update the EMA from.
        """

        with torch.no_grad():
            self.updates += 1
            decay = self.decay.fn(self.updates)

            # Get model state dict (de-parallelized)
            model_state_dict = de_parallel(model).state_dict()

            # Update EMA parameters
            for key, ema_val in self.ema.state_dict().items():
                if ema_val.dtype.is_floating_point:
                    ema_val *= decay
                    ema_val += (1.0 - decay) * model_state_dict[key].detach()

    def update_attr(self, model: torch.nn.Module, include: tuple = (), exclude: tuple = ('process_group', 'reducer')):
        """
        Updates EMA attributes from the model.

        Args:
            model (torch.nn.Module): The model to copy attributes from.
            include (tuple, optional): Attributes to explicitly include. Defaults to empty tuple.
            exclude (tuple, optional): Attributes to exclude. Defaults to ('process_group', 'reducer').
        """

        copy_attr(self.ema, model, include, exclude)

def weights_init(net: nn.Module, init_type: str = 'kaiming', init_gain: float = 0.02) -> None:
    """
    Initializes the weights of a PyTorch network.

    Args:
        net (torch.nn.Module): The network to initialize.
        init_type (str, optional): The weight initialization type. Defaults to 'normal'.
            - 'normal': Normal distribution initialization.
            - 'xavier': Xavier normal initialization.
            - 'kaiming': Kaiming normal initialization.
            - 'orthogonal': Orthogonal initialization.
        init_gain (float, optional): The gain factor for some initialization methods. Defaults to 0.02.
    """

    def init_fn(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                if classname.find('Conv') != -1:
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                else:
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print(f'Initializing network with {init_type} type')
    net.apply(init_fn)

def get_lr_scheduler(
    lr_decay_type: str,
    lr: float,
    min_lr: float,
    total_iters: int,
    warmup_iters_ratio: float = 0.05,
    warmup_lr_ratio: float = 0.1,
    no_aug_iter_ratio: float = 0.05,
    step_num: int = 10,
) -> Callable:
    """
    Gets a learning rate scheduler function based on the specified type.

    Args:
        lr_decay_type (str): The type of learning rate decay. Supported types are:
            - "cos": Cosine annealing with warmup and no-augmentation period.
            - other: Step decay.
        lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        total_iters (int): Total number of iterations.
        warmup_iters_ratio (float, optional): Ratio of warmup iterations to total iterations. Defaults to 0.05.
        warmup_lr_ratio (float, optional): Ratio of warmup learning rate to initial learning rate. Defaults to 0.1.
        no_aug_iter_ratio (float, optional): Ratio of no-augmentation iterations to total iterations. Defaults to 0.05.
        step_num (int, optional): Number of steps for step decay. Defaults to 10.

    Returns:
        Callable: A learning rate scheduler function that takes the current iteration as input and returns the learning rate.
        Raises ValueError if lr_decay_type is not supported.
    """

    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        """cosine annealing with warmup and no-aug period."""
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        """Step learning rate decay."""
        if step_size < 1:
            raise ValueError("step_size must be at least 1.")
        n = iters // step_size
        out_lr = lr * decay_rate**n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(int(warmup_iters_ratio * total_iters), 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(int(no_aug_iter_ratio * total_iters), 1), 15)
        return partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    elif lr_decay_type == "step":
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1)) if step_num > 1 else 0.0 # handle case where step_num=1
        step_size = total_iters / step_num
        return partial(step_lr, lr, decay_rate, step_size)
    else:
        raise ValueError(f"Unsupported lr_decay_type: {lr_decay_type}")

def set_optimizer_lr(optimizer: optim.Optimizer, lr_scheduler_func: Callable, epoch: int) -> None:
    """
    Sets the learning rate of the optimizer using a scheduler function.

    Args:
        optimizer (optim.Optimizer): The optimizer whose learning rate should be updated.
        lr_scheduler_func (Callable): A function that takes the current epoch/iteration
            and returns the desired learning rate.
        epoch (int): The current epoch or iteration number.
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
