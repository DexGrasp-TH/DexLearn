# 实习常用command

- 启动UR
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10 robot_ip:=192.168.1.102
```

- 开相机
```
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true pointcloud.ordered_pc:=true
```

- 测UR逆运动学
```
# Terminal 1
rviz2

# Terminal 2
ros2 run robot_state_publisher robot_state_publisher ~/ur10e.urdf

# Terminal 3
cd /home/peter/intern/ros2_ws/src/grasp_perception/scripts
python3 ./planning_node.py

# Terminal 4 (trigger ur move)
ros2 service call /move_ur10e std_srvs/Trigger
```

- LEAP Hand
```
ros2 launch leap_hand launch_leap.launch.py
```


# 实习成果演示（Jul 23）

- **Terminal 1:** 服务器`10.21.70.145`
```
cd ~/intern/grasp_synthesis/hardware
conda activate dexlearn
python ./graspgen_endpoint.py
```

- **Terminal 2:** 本地，打开相机数据流
```
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true pointcloud.ordered_pc:=true
```

- **Terminal 3:** 本地，打开点云分割
```
source ~/intern/ros2_ws/install/setup.bash
cd ~/intern/ros2_ws/src/grasp_perception/scripts
XDG_SESSION_TYPE=x11 python3 ./perception_node.py
```

- **Terminal 4:** 本地，发布静态坐标变换
```
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 camera_link world
```

- **Terminal 5:** 本地，运行抓取客户端
```
source ~/intern/ros2_ws/install/setup.bash
cd ~/intern/ros2_ws/src/grasp_perception/scripts
python3 ./main_grasp_client.py
```

- **Terminal 6:** 本地，运行可视化
```
source ~/intern/ros2_ws/install/setup.bash
ros2 launch lz_gripper_rviz display_dummy.launch.py
```
