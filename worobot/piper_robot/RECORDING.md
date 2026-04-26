# Tube Insertion / Cup Block Recording — woan + PiperRobot

This is the merged recording pipeline that replaces the kai0 collect script.

## Why this rewrite

The kai0 `collect_tube_insertion.py` had two bugs that made the previously
recorded dataset useless for training a working VLA:

1. **Action = state**. The HDF5→LeRobot converter set `action[t] = qpos[t]`
   instead of using the recorded master arm command. The model learned to
   predict the *achieved* joint positions (which lag the master), so the
   gripper output never reaches the full-close command magnitude → robot
   approaches the tube but cannot grip.
2. **Left/right naming flipped**. `action[0:7]` was assigned to the physical
   right arm and `action[7:14]` to the left arm; cameras `hand_left.mp4` and
   `hand_right.mp4` were also swapped. Trained model is internally consistent
   but mirror-flipped vs. the standard Piper convention.

The woan stack handles both correctly:
- `PiperRobot.get_observation()` returns slave qpos (state).
- `PiperRobot.get_action()` returns master qpos (true action target).
- LeRobotDataset writes `state` and `action` with their correct semantics.
- Camera key strings (`top_head`/`hand_left`/`hand_right`) follow the standard
  Piper convention so deployment needs no swap.

## What was patched on top of the upstream woan project

| File | Change |
|---|---|
| `worobot/piper_robot/piper_robot.py` | Rewritten on **ROS1 (rospy)** — subscribes to kai0 topics `/master/joint_{left,right}` (action) and `/puppet/joint_{left,right}` (state); publishes back on `/master/joint_*` only in non-teleop mode (replay/inference). Added `go_home()` calling `/can_{left,right}/go_zero_master_slave` + `restore_ms_mode` (`std_srvs/Trigger`). |
| `lerobot/common/utils/control_utils.py` | Keyboard listener now also accepts `c` (right pedal → SAVE) and `a` (left pedal → DISCARD/rerecord). Arrow keys still work. |
| `worobot/record.py` | After every episode (save **or** discard), automatically calls `robot.go_home()` before entering the reset wait. |

## Camera serial numbers (DO NOT CHANGE)

These match the kai0 hardware rig. Keep the launch script identical so the
data matches the exact viewpoints used in earlier collections.

Names follow the fold_clothes_data167 schema (`head`/`left`/`right`) so the
same deployment `camera_map` works for fold, tube, and cup_block models.

| Logical key | Physical role | RealSense serial |
|---|---|---|
| `head`  | front overhead (kai0 `cam_high`) | `233622079256` |
| `left`  | left arm wrist (kai0 `cam_left_wrist`) | `233722071228` |
| `right` | right arm wrist (kai0 `cam_right_wrist`) | `233622073364` |

## ROS1 prerequisites

This project uses ROS1 (rospy) and connects directly to the kai0 piper
master-slave nodes that you already have running. Required topics/services:

| Topic / Service | Direction | Purpose |
|---|---|---|
| `/master/joint_left`, `/master/joint_right` (`sensor_msgs/JointState`) | sub (record) / pub (replay) | action source (teleop master arm command) |
| `/puppet/joint_left`, `/puppet/joint_right` (`sensor_msgs/JointState`) | sub | state source (slave actual joint position) |
| `/can_{left,right}/go_zero_master_slave` (`std_srvs/Trigger`) | call | per-episode auto-homing |
| `/can_{left,right}/restore_ms_mode` (`std_srvs/Trigger`) | call | restore master-slave coupling after homing |

Just bring up the kai0 piper master-slave node the same way you used to
record before; no ROS2 packages are needed. **Cameras are NOT started via
ROS** — the woan record script opens RealSense devices directly via the
Python SDK using the serials baked into `run_record_tube_insertion.sh`.

## Usage

```bash
# 1. roscore + your kai0 piper master-slave node (the same one you've used).
roscore &
bash ~/kai0/train_deploy_alignment/dagger/agilex/tube_insertion/1_start_piper.sh
# (no separate camera node — they are opened by the record script itself)

# 2. Start recording.
bash worobot/piper_robot/run_record_tube_insertion.sh \
     heart666888/tube_insertion_v2_$(date +%m%d) \
     200
```

Controls during recording:

| Key | Action |
|---|---|
| → arrow / `c` (right pedal) | SAVE current episode, home arms, advance |
| ← arrow / `a` (left pedal)  | DISCARD current episode, home arms, rerecord |
| Esc | Stop recording (current episode is dropped) |

After every save or discard the script automatically calls `go_home()`
to bring both master and slave arms to zero.

## Cup block

Same flow, just change the prompt:

```bash
bash worobot/piper_robot/run_record_tube_insertion.sh \
     heart666888/cup_block_v2_$(date +%m%d) 200
# then edit the launch script PROMPT, or copy + adapt to a sibling
# run_record_cup_block.sh.
```

For cup block prompt use:
`Place the cup on the conveyor belt and toss the yellow block into the cup.`
