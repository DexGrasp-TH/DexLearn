import glob
from os.path import join as pjoin
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_test_dataloader
from dexlearn.network.models import *


def load_model(config: DictConfig) -> None:
    set_seed(config.seed)
    config.wandb.mode = "disabled"

    # load checkpoint
    save_ckpt_dir = pjoin(config.output_folder, config.wandb.id, "ckpts")
    if config.resume:
        all_ckpts = sorted(glob.glob(pjoin(save_ckpt_dir, "step_**.pth")))
        if len(all_ckpts) > 0:
            if config.ckpt is None:
                config.ckpt = all_ckpts[-1]
            else:
                config.ckpt = all_ckpts[-1].split("step_")[0] + f"step_{config.ckpt}.pth"

    model = eval(config.algo.model.name)(config.algo.model)

    # load ckpt if exists
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ckpt_iter = ckpt["iter"]
        print("loaded ckpt from", config.ckpt)
    else:
        print("Find no ckpt!")
        exit(1)

    model.to(config.device)
    model.eval()

    return model

def predict(config: DictConfig, model) -> None:
    set_seed(config.seed)
    config.wandb.mode = "disabled"
    test_loader = create_test_dataloader(config)
    logger = Logger(config)

    gengrasp_result = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            robot_pose, log_prob = model.sample(data, config.algo.test_grasp_num)

            # select top k predictions with higher log_prob
            topk_indices = torch.topk(log_prob, config.algo.test_topk, dim=1).indices
            batch_indices = (
                torch.arange(robot_pose.size(0))
                .unsqueeze(1)
                .expand(-1, config.algo.test_topk)
            )
            robot_pose = robot_pose[batch_indices, topk_indices]
            log_prob = log_prob[batch_indices, topk_indices]

            # detect tabletop setting
            if config.data.scene == "tabletop":
                assert "object_pose" in data
                robot_pose[..., :2] += data["object_pose"][:, :2].unsqueeze(1).unsqueeze(2)

            save_dict = {
                "pregrasp_qpos": robot_pose[..., 0, :],
                "grasp_qpos": robot_pose[..., 1, :],
                "squeeze_qpos": robot_pose[..., 2, :],
                "grasp_error": -log_prob,
                "scene_path": data["scene_path"],
            }

            gengrasp_result.append(save_dict)
            logger.save_samples(save_dict, 50000, data["save_path"])

    return gengrasp_result
