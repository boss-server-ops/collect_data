#!/usr/bin/env bash
# Record tube_insertion teleop episodes via woan + PiperRobot (ROS2).
#
# Camera serial numbers MUST match the kai0 setup that was used originally
# so that recorded data is identical to previous fold/piper datasets:
#   front (top_head)        = 233622079256
#   left  (hand_left wrist) = 233722071228
#   right (hand_right wrist)= 233622073364
#
# Controls:
#   Right arrow / 'c' (right pedal) → SAVE episode
#   Left  arrow / 'a' (left  pedal) → DISCARD + rerecord
#   Esc                              → STOP recording
# After every save or discard, robot.go_home() is called automatically
# (ROS2 service /can_{left,right}/go_zero_master_slave).
#
# Usage:
#   bash worobot/piper_robot/run_record_tube_insertion.sh [DATASET_REPO_ID] [NUM_EPISODES]
# Defaults:
#   DATASET_REPO_ID = heart666888/tube_insertion_v2_$(date +%m%d)
#   NUM_EPISODES    = 200

set -euo pipefail

DATASET_REPO_ID="${1:-heart666888/tube_insertion_v2_$(date +%m%d)}"
NUM_EPISODES="${2:-200}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/${DATASET_REPO_ID//\//_}}"

PROMPT="Pick up the test tube from the conveyor belt and insert it into the test tube rack."

# Camera serials (DO NOT CHANGE — must match recording rig).
# Camera key names match fold_clothes_data167 schema: head/left/right
# (no "top_" or "hand_" prefix), so trained models share the same camera_map
# at deployment.
SN_HEAD=233622079256   # front overhead (was "cam_high" in kai0)
SN_LEFT=233722071228   # left arm wrist (was "cam_left_wrist" in kai0)
SN_RIGHT=233622073364  # right arm wrist (was "cam_right_wrist" in kai0)

# Build cameras dict for tyro CLI
CAMERAS_JSON="{
  \"head\":  {\"type\":\"intelrealsense\",\"serial_number_or_name\":\"${SN_HEAD}\",\"fps\":30,\"width\":640,\"height\":480},
  \"left\":  {\"type\":\"intelrealsense\",\"serial_number_or_name\":\"${SN_LEFT}\",\"fps\":30,\"width\":640,\"height\":480},
  \"right\": {\"type\":\"intelrealsense\",\"serial_number_or_name\":\"${SN_RIGHT}\",\"fps\":30,\"width\":640,\"height\":480}
}"

echo "==========================================="
echo "  Recording: ${DATASET_REPO_ID}"
echo "  Episodes : ${NUM_EPISODES}"
echo "  Root     : ${DATASET_ROOT}"
echo "  Prompt   : ${PROMPT}"
echo "==========================================="

python3 -m worobot.record \
    --robot.type=piper_robot \
    --robot.teleop=true \
    --robot.cameras="${CAMERAS_JSON}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.single_task="${PROMPT}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10 \
    --dataset.video=true \
    --dataset.push_to_hub=false \
    --display_data=false \
    --play_sounds=true
