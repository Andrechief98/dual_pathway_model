#!/bin/bash

CONTAINER_NAME="andrea_docker-ros1_noetic-1"
DURATION=40

BAG_NAME="VLM_3_obj"  
TOPICS="/vlm/inference/time" 
DEST_FOLDER="/home/ros_ws/src/dual_pathway_model/bag_files/real_world/VLM_latency"

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

OBJ_LIST="[\\\"rover\\\",\\\"cardboard_box\\\",\\\"person\\\"]"

# Thalamus
open_ros_window "THALAMUS" "rosrun low_road thalamus.py"
sleep 2

open_ros_window "RECORDER" "cd $DEST_FOLDER && rosbag record -O ${BAG_NAME}.bag $TOPICS"
open_ros_window "VLM" "roslaunch high_road test_VLM_latency.launch objects:=$OBJ_LIST"

sleep $DURATION

echo "End of the experiement. Closing everything ..."

# Killing all ROS processes
docker exec $CONTAINER_NAME /bin/bash -c "source /opt/ros/noetic/setup.bash && rosnode kill -a; killall -9 rosmaster gzserver"

# Killing 
pkill -9 -f gzclient

