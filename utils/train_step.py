import os
import torch
from tqdm import tqdm

from utils.utils import get_lr


def train_one_epoch(model_train, model, ema, model_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, epochs, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    """
    Trains the model for one epoch.

    Args:
        model_train (torch.nn.Module): Model to be trained.
        model (torch.nn.Module): Model for evaluation (ema or base model).
        ema (optional, torch.nn.Module): EMA model for evaluation.
        model_loss (torch.nn.Module): Loss function.
        loss_history (object): Object to store training history.
        eval_callback (object): Callback for evaluation.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch.
        epoch_step (int): Number of training steps per epoch.
        epoch_step_val (int): Number of validation steps per epoch.
        gen (data.DataLoader): Training data loader.
        gen_val (data.DataLoader): Validation data loader.
        epochs (int): Total number of epochs.
        cuda (bool): Whether to use CUDA.
        fp16 (bool): Whether to use mixed precision training.
        scaler (torch.cuda.amp.GradScaler): GradScaler for mixed precision.
        save_period (int): How often to save weights.
        save_dir (str): Directory to save weights.
        local_rank (int, optional): Local rank for distributed training. Defaults to 0.
    """

    total_loss = 0
    val_loss = 0

    if local_rank == 0:
        print(f'Start Train - Epoch {epoch + 1}/{epochs}')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{epochs}')

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, bboxes = batch

        if cuda:
            images = images.to(local_rank)
            bboxes = bboxes.to(local_rank)

        optimizer.zero_grad()

        if not fp16:
            # Forward pass
            outputs = model_train(images)
            loss_value = model_loss(outputs, bboxes)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model_train(images)
                loss_value = model_loss(outputs, bboxes)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        total_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix({'loss': total_loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epochs}')

    if ema:
        model_eval = ema.ema
    else:
        model_eval = model.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images, bboxes = batch

        if cuda:
            images = images.to(local_rank)
            bboxes = bboxes.to(local_rank)

        with torch.no_grad():
            # Forward pass (validation)
            outputs = model_eval(images)
            loss_value = model_loss(outputs, bboxes)

        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix({'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(epochs))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # Save weights
        save_state_dict = ema.ema.state_dict() if ema else model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == epochs:
            torch.save(save_state_dict, os.path.join(save_dir, f"ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth"))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))