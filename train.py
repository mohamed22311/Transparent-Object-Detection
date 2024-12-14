#-------------------------------------#
#   Train the dataset
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom model (BaseModel)
from nn import BaseModel  # Import the BaseModel architecture
from nn import Loss, ModelEMA, get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import CustomDataset, custom_dataset_collate  # Replace with your dataset and collate function
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.train_step import train_one_epoch

'''
When training your own model, ensure the following:
1. Verify dataset format:
   - Input images: .jpg format, no fixed size (resized during training).
   - Grayscale images are automatically converted to RGB.
   - Non-.jpg images should be converted to .jpg before training.
   - Labels: .xml format, containing object information corresponding to each image.

2. Loss value trends:
   - Loss should decrease over epochs to indicate convergence.
   - Loss values are relative and depend on the loss function; they don't need to approach zero.
   - Training loss logs are saved in the `logs` folder.

3. Model weights:
   - Saved in the `logs` folder after each epoch.
   - Ensure epoch continuity when resuming training by adjusting `Init_Epoch` and `Freeze_Epoch`.
'''

if __name__ == "__main__":
    #---------------------------------#
    #   Cuda: Use GPU if available
    #---------------------------------#
    Cuda = True

    #---------------------------------#
    #   Seed: Fix random seed for reproducibility
    #---------------------------------#
    seed = 11

    #---------------------------------#
    #   Distributed Training: Enable multi-GPU training
    #   DP Mode: Use DataParallel (Windows default)
    #   DDP Mode: Use DistributedDataParallel (Linux recommended)
    #---------------------------------#
    distributed = False

    #---------------------------------#
    #   Sync BatchNorm: Use synchronized batch normalization for DDP
    #---------------------------------#
    sync_bn = False

    #---------------------------------#
    #   Mixed Precision (FP16): Reduce memory usage (requires PyTorch 1.7.1+)
    #---------------------------------#
    fp16 = False

    #---------------------------------#
    #   Classes Path: Path to the class names file
    #---------------------------------#
    classes_path = 'model_data/voc_classes.txt'

    #---------------------------------#
    #   Model Path: Path to pre-trained weights
    #   - Use pre-trained weights for better performance.
    #   - Set to '' to train from scratch.
    #---------------------------------#
    model_path = 'model_data/focus_pretrained.pth'  # Replace with your model's pre-trained weights

    #---------------------------------#
    #   Input Shape: Input image size (must be divisible by 32)
    #---------------------------------#
    input_shape = [640, 640]

    #---------------------------------#
    #   Model Parameters:
    #   - base_channels: Base number of channels in the model
    #   - base_depth: Base depth of the model
    #   - deep_mul: Depth multiplier for the model
    #---------------------------------#
    base_channels = 1024
    base_depth = 4
    deep_mul = 1.0

    #---------------------------------#
    #   Pretrained: Use pre-trained backbone weights
    #   - Ignored if `model_path` is set.
    #---------------------------------#
    pretrained = False

    #---------------------------------#
    #   Data Augmentation:
    #   - Mosaic: Enable mosaic augmentation (default: 50% probability)
    #   - Mixup: Enable mixup augmentation (default: 50% probability)
    #   - Special Augmentation: Apply mosaic for the first 70% of epochs
    #---------------------------------#
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7

    #---------------------------------#
    #   Label Smoothing: Smooth labels to prevent overfitting
    #---------------------------------#
    label_smoothing = 0

    #---------------------------------#
    #   Training Parameters:
    #   - Freeze Training: Freeze backbone for initial training
    #   - Batch Sizes: Adjust based on GPU memory
    #   - Epochs: Total training epochs
    #---------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 32
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = 16
    Freeze_Train = True

    #---------------------------------#
    #   Optimization Parameters:
    #   - Learning Rate: Initial and minimum learning rates
    #   - Optimizer: Adam or SGD
    #   - Momentum: Momentum for SGD
    #   - Weight Decay: L2 regularization
    #   - LR Decay Type: Step or Cosine decay
    #---------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = "cos"

    #---------------------------------#
    #   Checkpoint and Logging:
    #   - Save Period: Save model every N epochs
    #   - Save Directory: Directory for logs and weights
    #   - Evaluation: Enable evaluation on validation set
    #---------------------------------#
    save_period = 10
    save_dir = 'logs'
    eval_flag = True
    eval_period = 10

    #---------------------------------#
    #   Data Loading:
    #   - Num Workers: Number of threads for data loading
    #---------------------------------#
    num_workers = 4

    #---------------------------------#
    #   Dataset Paths:
    #   - Train and validation annotation files
    #---------------------------------#
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    #---------------------------------#
    #   Seed Everything: Fix random seed for reproducibility
    #---------------------------------#
    seed_everything(seed)

    #---------------------------------#
    #   Device Setup:
    #   - Use GPU if available
    #---------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("GPU Device Count:", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    #---------------------------------#
    #   Load Class Names and Number of Classes
    #---------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #---------------------------------#
    #   Download Pretrained Weights
    #---------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(model_path)  # Replace with your download function
            dist.barrier()
        else:
            download_weights(model_path)

    #---------------------------------#
    #   Create Custom Model (BaseModel)
    #---------------------------------#
    model = BaseModel(num_classes, base_channels, base_depth, deep_mul).to(device)  # Replace with your model class

    #---------------------------------#
    #   Load Pretrained Weights
    #---------------------------------#
    if model_path != '':
        if local_rank == 0:
            print(f'Loading weights from {model_path}.')

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print(f"Loaded keys: {load_key[:500]}... ({len(load_key)} keys)")
            print(f"Failed to load keys: {no_load_key[:500]}... ({len(no_load_key)} keys)")
            print("\033[1;33;44mNote: Missing head weights is normal; missing backbone weights is an error.\033[0m")

    #---------------------------------#
    #   Define Loss Function
    #---------------------------------#
    custom_loss = Loss(model)  # Replace with your custom loss function

    #---------------------------------#
    #   Initialize Loss History
    #---------------------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    #---------------------------------#
    #   Mixed Precision Setup
    #---------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    #---------------------------------#
    #   Model Training Setup
    #---------------------------------#
    model_train = model.train()

    #---------------------------------#
    #   Sync BatchNorm for Distributed Training
    #---------------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported in single-GPU or non-distributed mode.")

    #---------------------------------#
    #   Multi-GPU Setup
    #---------------------------------#
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------------#
    #   Exponential Moving Average (EMA)
    #---------------------------------#
    ema = ModelEMA(model_train)

    #---------------------------------#
    #   Load Dataset Annotations
    #---------------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

        #---------------------------------#
        #   Training Step Calculation
        #---------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset is too small to train. Please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(f"\033[1;33;44m[Warning] Using {optimizer_type}, recommended total steps: {wanted_step}.\033[0m")
            print(f"\033[1;33;44m[Warning] Current total steps: {total_step}. Recommended epochs: {wanted_epoch}.\033[0m")

    #---------------------------------#
    #   Freeze Training Setup
    #---------------------------------#
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #---------------------------------#
        #   Learning Rate Adjustment
        #---------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------#
        #   Optimizer Setup
        #---------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------#
        #   Learning Rate Scheduler
        #---------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        #---------------------------------#
        #   Dataset Loader Setup
        #---------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to train. Please expand the dataset.")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        train_dataset = CustomDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                      mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
                                      train=True, special_aug_ratio=special_aug_ratio)
        val_dataset = CustomDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                    mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                    special_aug_ratio=0)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=custom_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=custom_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #---------------------------------#
        #   Evaluation Callback
        #---------------------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda,
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        #---------------------------------#
        #   Training Loop
        #---------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset is too small to train. Please expand the dataset.")

                if ema:
                    ema.updates = epoch_step * epoch

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=custom_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     drop_last=True, collate_fn=custom_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            train_one_epoch(model_train, model, ema, custom_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()