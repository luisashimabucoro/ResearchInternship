# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer
import argparse


def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    config = Config('/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/LibFewShot/reproduce/Baseline/miniImageNet.yaml').get_config_dict()
    # config = Config('/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/LibFewShot/reproduce/Proto/cross_domain.yaml').get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)