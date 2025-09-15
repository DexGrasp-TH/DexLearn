import torch
from einops import rearrange, repeat

from .diffusion_util import MLPWrapperNoTime
from dexlearn.utils.RMS import Normalization


class GraspEvaluator_MLPRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = (12 + 16) * 3  # (pregrasp, grasp, squeeze) qpos, 12 for (3x3) rot & 3 pos, 16 for leap hand
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=1,
            act="mish",
        )
        self.head = MLPWrapperNoTime(
            channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters
        )
        self.rms = True
        if self.rms:
            self.RMS = Normalization(N_out)
        self.sigmoid = torch.nn.Sigmoid()
        self.pred_loss = torch.nn.BCEWithLogitsLoss()
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")  # (256, 1, 3, 3) -> (256, 3, 3)
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")  # (256, 1, 3, 3, 3) -> (256, 3, 3, 3)
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) n x")  # (256, 1, 3, 16) -> (256, 3, 16)
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rt = torch.cat(
            [repeat(hand_rot, "b n x y -> b n (x y)"), hand_trans, hand_joint], dim=-1
        )   # (256, 3, 28)
        grasp_rt = rearrange(grasp_rt, "b n x -> b (n x)")  # (256, 84)
        
        if self.rms:
            grasp_rt_diff = self.RMS(grasp_rt)
        else:
            grasp_rt_diff = grasp_rt
            
        pred_prob = self.sigmoid(self.head(grasp_rt_diff, global_feature))
            
        result_dict["loss_evaluation"] = self.pred_loss(pred_prob, data["success"].float())
        with torch.no_grad():
            result_dict["num_success"] = (
                ((pred_prob > 0.5).float() == data["success"].float()).sum() / batch_num
            )
        
        return result_dict