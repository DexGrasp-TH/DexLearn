import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Predictor(object):
    def __init__(self):
        self.initialize_sam2()

    def initialize_sam2(self):
        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        # build the SAM2 model
        sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        predictor = SAM2ImagePredictor(sam2_model)
        self.predictor = predictor

    def run_sam2(self, image: np.ndarray, clicks: list) -> np.ndarray:
        """
        Run SAM2 inference on the given image with the provided clicks.

        :param image: Input image as a numpy array.
        :param clicks: List of click coordinates.
        :return: Mask as a numpy array.
        """
        # Convert image to PIL format
        # image_pil = Image.fromarray(image)
        # self.predictor.set_image(image_pil)
        self.predictor.set_image(image)

        # Point and label
        input_point = np.array(clicks, dtype=np.float32)
        input_label = np.ones((len(clicks),), dtype=int)

        # Run SAM2 inference
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        result = {
            "masks": masks,
            "scores": scores,
            "logits": logits
        }

        return result
