# 环境创建
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

conda create -y -n mini_lerobot python=3.10
conda activate mini_lerobot
conda install ffmpeg=7.0 -c conda-forge
```

# 依赖安装
```
# lerobot配置
pip install -e .

pip uninstall -y datasets
pip install datasets==3.0

#采数据依赖
pip install pyrealsense2
pip install fastapi uvicorn[standard]
pip install flask==3.1.2
#模型训练依赖
pip install grpcio grpcio-tools==1.74.0
pip install transformers==4.51.3
pip install num2words==0.5.14
pip install accelerate==1.10.0
```

# 采集数据
```
0、确认机械臂开启
ros2 topic list 查看相关消息是否存在
x1_robot：
自启动，可直接查看相关消息
piper_robot：
bash start_piper.sh #具体请参考piper ros

1、采集机械臂数据
x1_robot：
移动脚本到x1_robot小脑A路径
cp woanlerobot/worobot/x1_robot/record_action.py A路径
source /opt/ros/humble/setup.bash
python3 record_action.py
piper_robot：
cd woanlerobot/worobot/x1_robot
source /opt/ros/humble/setup.bash
python3 record_action.py

2、数据录制
conda activate mini_lerobot
bash start_record.sh
注意：修改 --robot.type x1_robot 、--robot.id my_x1_robot 参数

3、images转mp4
bash start_encode_images.sh

```

# 查看数据
```
conda activate mini_lerobot
bash start_view.sh
```

# 回放数据
```
conda activate mini_lerobot
bash start_replay.sh
```