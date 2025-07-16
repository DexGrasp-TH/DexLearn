import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torch.utils.data._utils.collate import default_collate
from omegaconf import DictConfig, ListConfig
import MinkowskiEngine as ME
import torch
from copy import deepcopy

from .base_dex import DexDataset


def create_dataset(config, mode):
    sp_voxel_size = (
        config.algo.model.backbone.voxel_size
        if "MinkUNet" in config.algo.model.backbone.name
        else None
    )
    if isinstance(config.data.object_path, ListConfig):
        dataset_lst = []
        for p in config.data.object_path:
            new_data_config = deepcopy(config.data)
            new_data_config.object_path = p
            dataset_lst.append(DexDataset(new_data_config, mode, sp_voxel_size))
        dataset = torch.utils.data.ConcatDataset(dataset_lst)
    else:
        dataset = DexDataset(config.data, mode, sp_voxel_size)
    return dataset


def create_train_dataloader(config: DictConfig):
    train_dataset = create_dataset(config, mode="train")
    val_dataset = create_dataset(config, mode="eval")

    train_loader = InfLoader(
        DataLoader(
            train_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=config.data.num_workers,
            shuffle=True,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )
    val_loader = InfLoader(
        DataLoader(
            val_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=config.data.num_workers,
            shuffle=False,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )
    return train_loader, val_loader


def create_test_dataloader(config: DictConfig, mode="test"):
    test_dataset = create_dataset(config, mode=mode)
    test_loader = FiniteLoader(
        DataLoader(
            test_dataset,
            batch_size=config.algo.batch_size,
            drop_last=False,
            num_workers=config.data.num_workers,
            shuffle=False,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )
    return test_loader


class InfLoader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.iter_loader = iter(self.loader)
        self.device = device

    def get(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            data = next(self.iter_loader)

        for k, v in data.items():
            if type(v).__module__ == "torch":
                if (
                    "Int" not in v.type()
                    and "Long" not in v.type()
                    and "Short" not in v.type()
                ):
                    v = v.float()
                data[k] = v.to(self.device)
        return data


class FiniteLoader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.iter_loader = iter(self.loader)
        self.device = device

    def __len__(self):
        return len(self.iter_loader)

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.iter_loader)
        for k, v in data.items():
            if type(v).__module__ == "torch":
                if (
                    "Int" not in v.type()
                    and "Long" not in v.type()
                    and "Short" not in v.type()
                ):
                    v = v.float()
                data[k] = v.to(self.device)
        return data


# some magic to get MinkowskiEngine sparse tensor
def minkowski_collate_fn(list_data):
    scene_path_data = None
    if "scene_path" in list_data[0].keys():
        scene_path_data = [d.pop("scene_path") for d in list_data]

    coors_data = None
    if "coors" in list_data[0].keys():
        coors_data = [d.pop("coors") for d in list_data]
        feats_data = [d.pop("feats") for d in list_data]
        coordinates_batch, features_batch = ME.utils.sparse_collate(
            coors_data, feats_data
        )
        coordinates_batch, features_batch, original2quantize, quantize2original = (
            ME.utils.sparse_quantize(
                coordinates_batch,
                features_batch,
                return_index=True,
                return_inverse=True,
            )
        )

    res = default_collate(list_data)

    if scene_path_data is not None:
        res["scene_path"] = scene_path_data

    if coors_data is not None:
        res["coors"] = coordinates_batch
        res["feats"] = features_batch
        res["original2quantize"] = original2quantize
        res["quantize2original"] = quantize2original
    return res


def get_sparse_tensor(pc: torch.tensor, voxel_size: float):
    """
    pc: (B, N, 3)
    return dict(point_clouds, coors, feats, quantize2original)
    """
    coors = pc / voxel_size
    feats = pc
    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [coor for coor in coors], [feat for feat in feats]
    )
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(),
        features_batch,
        return_index=True,
        return_inverse=True,
    )
    return dict(
        point_clouds=pc,
        coors=coordinates_batch.to(pc.device),
        feats=features_batch,
        quantize2original=quantize2original.to(pc.device),
    )


if __name__ == "__main__":
    import yaml
    from omegaconf import OmegaConf
    from base_dex import DexDataset

    from glob import glob
    import numpy as np

    path_lst = glob('assets/grasp/bodex_shadow_tabletop/succ_collect/1_TypeUnaware/**/**/**.npy', recursive=True)

    invalid_data_num = 0
    for path in path_lst:
        data = np.load(path, allow_pickle=True).item()
        if data['grasp_qpos'].shape[1] == 18:
            # print(path)
            invalid_data_num += 1

    print(f"Invalid data num: {invalid_data_num}")
    
    sp_voxel_size = 0.005

    config = yaml.safe_load(open('/home/jyp/research/DexLearn/dexlearn/config/data/bodex_shadow_tabletop.yaml', 'r'))
    config['grasp_path'] = config['grasp_path'].replace('${data_folder}', 'assets')
    config['object_path'] = config['object_path'].replace('${data_folder}', 'assets')
    config = OmegaConf.create(config)

    train_dataset = DexDataset(config, "train", sp_voxel_size)

    loader = DataLoader(
            train_dataset,
            batch_size=256,
            drop_last=True,
            num_workers=0,
            shuffle=True,
            collate_fn=minkowski_collate_fn,
        )
    for data in loader:
        breakpoint()

    train_loader = InfLoader(
        DataLoader(
            train_dataset,
            batch_size=256,
            drop_last=True,
            num_workers=16,
            shuffle=True,
            collate_fn=minkowski_collate_fn,
        ),
        'cuda:0',
    )

    for it in range(100):
        print(f"Iteration {it}")
        data = train_loader.get()
        breakpoint()
