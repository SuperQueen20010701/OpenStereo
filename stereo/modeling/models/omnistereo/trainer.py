# @Time    : 2024/2/9 11:39, 2026/1/19 16:58
# @Author  : zhangchenming, Qian Zhou
import time
import contextlib

import torch
import torch.distributed as dist
from stereo.utils.lamb import Lamb
from stereo.utils import common_utils
from stereo.utils.warmup import LinearWarmup
from stereo.modeling.trainer_template import TrainerTemplate
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from .core.omnistereo import OmniStereo

import logging

logger = logging.getLogger('OmnistereoTrainer')

__all__ = {
    'OmniStereo': OmniStereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        self.accumulation_steps = cfgs.OPTIMIZATION.get('GRADIENT_ACCUMULATION_STEPS', 1) # set the Gradient Accumulation (8)
        if self.accumulation_steps < 1:
            self.accumulation_steps = 1
        
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

        # ---- precision config ----
        # AMP: True/False
        self.amp_enabled = bool(self.cfgs.OPTIMIZATION.AMP)
        # AMP_DTYPE: "bf16" / "fp16" (default fp16)
        self.amp_dtype_cfg = str(self.cfgs.OPTIMIZATION.get("AMP_DTYPE", "fp16")).lower()
        self.use_bf16 = self.amp_enabled and (self.amp_dtype_cfg in ["bf16", "bfloat16"])
        self.use_fp16 = self.amp_enabled and (not self.use_bf16)
        
        # no scaler for bf16, only for fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
     
    def build_optimizer_and_scheduler(self):
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name'])
        optimizer = optimizer_cls(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)

        accumulation_steps = self.accumulation_steps
        effective_optimizer_steps = self.max_iter // max(accumulation_steps, 1)
        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = effective_optimizer_steps
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        valid_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER, ['name', 'on_epoch'])
        if self.cfgs.OPTIMIZATION.SCHEDULER.NAME == "CosineAnnealingLR":
            valid_arg["T_max"] = effective_optimizer_steps
        scheduler = scheduler_cls(optimizer, **valid_arg)

        return optimizer, scheduler
        
    def build_warmup(self):
        effective_step_per_epoch = len(self.train_loader) // self.accumulation_steps
        last_step = (self.last_epoch + 1) * effective_step_per_epoch - 1
        if 'WARMUP' in self.cfgs.OPTIMIZATION.SCHEDULER:
            raw_warmup_steps = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARM_STEPS', 1)
            warmup_steps = max(1,raw_warmup_steps // self.accumulation_steps)
            lr_ratio = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARMUP_LR_RATIO', 0.0)
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=warmup_steps,
                last_step=last_step,
                warmup_lr_ratio=lr_ratio)
        else:
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=1,
                last_step=last_step)

        return warmup_scheduler

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL

        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss
        self.optimizer.zero_grad(set_to_none=True)

        epoch_total_loss = 0.0
        epoch_valid_iter_count = 0
        accumulation_counter_per_epoch = 0
        optimizer_step_count = 0
        accumulation_steps = self.accumulation_steps

        if self.local_rank == 0:
            self.logger.info(f'Using gradient accumulation with {accumulation_steps} steps. '
                           f'Effective batch size per GPU: {self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU * accumulation_steps}') # gradient accumulation steps * batch size per GPU
        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            if total_iter >= self.max_iter:
                break
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()
            
            with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            is_invalid = torch.isnan(loss) | torch.isinf(loss) | (loss < 1e-6) # loss
            invalid_flag = torch.tensor([1 if is_invalid else 0], dtype=torch.int, device=loss.device)

            if self.args.dist_mode and dist.is_available() and dist.is_initialized():
                dist.all_reduce(invalid_flag, op=dist.ReduceOp.MAX) # boardcast invalid flag to all processes

            global_invalid = invalid_flag.item() > 0
            if global_invalid:
                if self.local_rank == 0:
                    self.logger.warning(
                        f'loss have nan/inf at iter {i}, epoch {current_epoch}, '
                        f'rank {self.local_rank}, continue~'
                    )
                del model_pred, loss
                torch.cuda.empty_cache()
                continue

            epoch_valid_iter_count += 1
            accumulation_counter_per_epoch += 1
            is_last_accumulation = (accumulation_counter_per_epoch == accumulation_steps) or (i == len(self.train_loader) - 1) # last accumulation step or last iteration of the epoch
            curr_accum_counter = accumulation_counter_per_epoch if is_last_accumulation else accumulation_steps
            scaled_loss = loss / curr_accum_counter
            context = self.model.no_sync() if (self.args.dist_mode and not is_last_accumulation) else contextlib.nullcontext()
            with context:
                if self.use_fp16:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            # last accumulation step -> use optimizer update 
            if is_last_accumulation:
                try:
                    if self.use_fp16:
                        self.scaler.unscale_(self.optimizer)
                        if self.clip_gard is not None:
                            self.clip_gard(self.model)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.clip_gard is not None:
                            self.clip_gard(self.model)
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    # Update learning rate scheduler only when optimizer steps
                    with self.warmup_scheduler.dampening():
                        if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                            self.scheduler.step()
                    accumulation_counter_per_epoch = 0
                    optimizer_step_count += 1
                    
                    # Only accumulate loss for valid iterations (after successful optimizer step)
                    epoch_total_loss += loss.item()
                except Exception as e:
                    # If step fails, log and skip this iteration
                    if self.local_rank == 0:
                        self.logger.error(
                            f'[Rank {self.local_rank}] Error during optimizer step at iter {i}, epoch {current_epoch}: {e}. '
                            f'Skipping this iteration.'
                        )
                    # Clear gradients and continue
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulation_counter_per_epoch = 0
                    # Decrement valid iter count since this iteration failed
                    epoch_valid_iter_count -= 1
                    del model_pred, loss, scaled_loss
                    torch.cuda.empty_cache()
                    continue

            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            
            
            if is_last_accumulation >0 and accumulation_counter_per_epoch % logger_iter_interval == 0:
                # Calculate average loss over valid iterations (not skipped)
                avg_loss = epoch_total_loss / max(epoch_valid_iter_count, 1) # Avoid div by zero
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), avg_loss, lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION and is_last_accumulation:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            if self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)
                
            del model_pred, loss, scaled_loss

        # for the last epoch summary (only at the end of epoch)
        if self.local_rank == 0:
            avg_loss = epoch_total_loss / max(epoch_valid_iter_count, 1)
            self.logger.info(
                f"Epoch {current_epoch} finished. "
                f"AvgLoss={avg_loss:.6f}")