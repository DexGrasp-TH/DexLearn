import sys
import socket
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import numpy as np

from dexlearn.dataset.base_dex import convert_pc_to_network_input
from graspgen_predictor import load_model, predict


# global config
HOST = '10.21.70.145'
PORT = 50008

def process(cfg, model, points: np.ndarray, array: np.ndarray) -> np.ndarray:
    result = predict(cfg, model)[0]
    print(f"Prediction result: {result['pregrasp_qpos'][0]}")

    # convert torch to numpy
    for key in ['pregrasp_qpos', 'grasp_qpos', 'squeeze_qpos', 'grasp_error']:
        result[key] = result[key].cpu().numpy()

    return result

def main_func(cfg: DictConfig):
    model = load_model(cfg)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print("Server listening on", PORT)

        while True:
            conn, addr = s.accept()
            with conn:
                print("Connected by", addr)
                # 先读取数据长度
                data_len = int.from_bytes(conn.recv(8), 'big')
                data = b''
                while len(data) < data_len:
                    data += conn.recv(data_len - len(data))

                # 解析数据
                payload = pickle.loads(data)
                object_pcd = payload['points']
                object_pose = payload['array']

                # ONLY FOR DEBUG!
                # object_pcd = np.load('/home/hand/intern/DexLearn/assets/object/kitchen/vision_data/azure_kinect_dk/sem_MilkCarton_f5b5a24adc6826ace41b639931f9ca1/tabletop_ur10e/scale006_pose003_0/partial_pc_00.npy')
                # scene_cfg = np.load('/home/hand/intern/DexLearn/assets/object/kitchen/scene_cfg/sem_MilkCarton_f5b5a24adc6826ace41b639931f9ca1/tabletop_ur10e/scale006_pose003_0.npy', allow_pickle=True).item()
                # object_pose = scene_cfg['scene']['sem_MilkCarton_f5b5a24adc6826ace41b639931f9ca1']['pose']
                extras = {
                    "asset_root": "/home/hand/intern/DexLearn/assets/object/online",
                    "name": "example_object",
                    "pose": object_pose
                }
                convert_pc_to_network_input(object_pcd, extras)

                # convert format
                result = process(cfg, model, object_pcd, object_pose)

                print(f"Received point cloud P with shape {object_pcd.shape}, P[0]: {object_pcd[0]}")
                print(f"Received object pose {object_pose}")

                # 序列化结果并发送
                result_bytes = pickle.dumps(result)
                conn.sendall(len(result_bytes).to_bytes(8, 'big'))
                conn.sendall(result_bytes)


if __name__ == "__main__":
    # model_name = "bodex_lz_gripper_online_nflow_full"
    model_name = "bodex_leap_online_nflow_full"
    sys.argv = (
        sys.argv[:1]
        + list(OmegaConf.load(f"/home/hand/intern/DexLearn/output/{model_name}/.hydra/overrides.yaml"))
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
