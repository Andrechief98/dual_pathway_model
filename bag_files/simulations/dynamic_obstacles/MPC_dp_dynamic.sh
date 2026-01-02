#!/bin/bash

CONTAINER_NAME="andrea_docker-ros1_noetic-1"
DURATION=30

BAG_NAME="MPC_dp_dynamic"  
TOPICS="/odom /fearlevel /cmd_vel /thalamus/info /gazebo/model_states /mpc/params /mpc/statistics /amygdala/lowroad/risks /amygdala/highroad/risks /odometry/filtered" 
DEST_FOLDER="/home/ros_ws/src/dual_pathway_model/bag_files/simulations/dynamic_obstacles"

MSG='{
  header: {seq: 0, stamp: {secs: 0, nsecs: 0}, frame_id: "map"},
  pose: {
    position: {x: 10.0, y: 0.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'

# Function for opening a new window within the container
open_ros_window() {
    local TITLE=$1
    local COMMAND=$2
    
    gnome-terminal --title="$TITLE" -- bash -c "
        docker exec -it $CONTAINER_NAME bash -c \"
            source /opt/ros/noetic/setup.bash && \
            source /home/ros_ws/devel/setup.bash && \
            $COMMAND
        \"; exec bash" & 
}

# 1. Simulation - amygdala
open_ros_window "SIMULATION" "roslaunch mpc_planner test_architecture_dynamic.launch use_warm_start:=true high_road_influence:=0.5 test:='dp'"
sleep 10

# 2. Thalamus
open_ros_window "THALAMUS" "rosrun low_road thalamus.py"
sleep 5

# 3. VLM
open_ros_window "VLM" "rosrun high_road VisualLanguageModel.py"
sleep 6

# 4. Rosbag recorder
open_ros_window "RECORDER" "cd $DEST_FOLDER && rosbag record -O ${BAG_NAME}.bag $TOPICS"

# 5. Starting experiment (publishing robot's goal and obstacles controllers)
open_ros_window "OBSTACLES_CONTROLLER" "rosrun low_road controller_obstacles.py"
open_ros_window "GOAL_PUBLISHER" "rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '$MSG'"

sleep $DURATION

echo "End of the experiement. Closing everything ..."

# Killing all ROS processes
docker exec $CONTAINER_NAME /bin/bash -c "source /opt/ros/noetic/setup.bash && rosnode kill -a; killall -9 rosmaster gzserver"

# Killing Gazebo
pkill -9 -f gzclient

