import sys
import os
import argparse

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_test_dataloader
from dexlearn.network.models import *


def main_func(config: DictConfig) -> None:
    set_seed(config.seed)
    config.wandb.mode = "disabled"
    logger = Logger(config)
    test_loader = create_test_dataloader(config)

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

    # training
    model.to(config.device)
    model.eval()

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

            logger.save_samples(save_dict, ckpt_iter, data["save_path"])

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        required=True,
        help="experiment folder, e.g. shadow_tabletop_debug",
    )
    args, unknown = parser.parse_known_args()

    sys.argv = (
        sys.argv[:1]
        + unknown
        + list(OmegaConf.load(f"output/{args.exp_name}/.hydra/overrides.yaml"))
    )

    # remove duplicated args. Note: cmd has the priority!
    check_dict = {}
    for argv in sys.argv[1:]:
        arg_key = argv.split("=")[0]
        if arg_key not in check_dict:
            check_dict[arg_key] = True
        else:
            sys.argv.remove(argv)

    hydra.main(config_path="config", config_name="base", version_base=None)(main_func)()
