
import wandb 
from os.path import join as pjoin
import torch 
import os 
import glob 
import numpy as np

class Logger:
    def __init__(self, cfg):
        self.config = cfg.wandb
        self.save_ckpt_dir = pjoin(cfg.output_folder, cfg.exp_folder, self.config.id, 'ckpts')
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        self.save_test_dir = pjoin(cfg.output_folder, cfg.exp_folder, self.config.id, 'tests')
        os.makedirs(self.save_test_dir, exist_ok=True)
        
        wandb_resume = None
        if self.config.resume:
            all_ckpts = sorted(glob.glob(pjoin(self.save_ckpt_dir, 'step_**.pth')))
            if len(all_ckpts) > 0:
                wandb_resume = "allow"
                if cfg.ckpt is None:
                    cfg.ckpt = all_ckpts[-1]
                else:
                    cfg.ckpt = all_ckpts[-1].split('step_')[0] + f'step_{cfg.ckpt}.pth'
        
        wandb.init(
            dir=self.config.folder,
            project=self.config.project,
            group=self.config.group,
            id=self.config.id,
            mode=self.config.mode,
            resume=wandb_resume,     
        )

    def log(self, dic: dict, mode: str, step: int):
        """
            log a dictionary, requires all values to be scalar
            mode is used to distinguish train, val, ...
            step is the iteration number
        """
        wandb.log({f'{mode}/{k}': v for k, v in dic.items()}, step=step)
    
    def save(self, dic: dict, step: int):
        """
            save a dictionary to a file
        """
        torch.save(dic, pjoin(self.save_ckpt_dir, f'step_{str(step).zfill(6)}.pth'))
    
    def save_samples(self, dic: dict, step: int, save_path: list):
        for i, suffix in enumerate(save_path):
            save_dict = {}
            for k, v in dic.items():
                if type(v).__module__ == 'torch':
                    save_dict[k] = v[i].detach().cpu().numpy()
                else:
                    save_dict[k] = v[i]
            path = pjoin(self.save_test_dir, f'step_{str(step).zfill(6)}', suffix)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, save_dict)
            