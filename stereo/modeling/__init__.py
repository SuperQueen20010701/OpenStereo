# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming
from .models.omnistereo.trainer import Trainer as OmniStereoTrainer

__all__ = {
    'OmniStereo': OmniStereoTrainer,
}


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer)
    return trainer
