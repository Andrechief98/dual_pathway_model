#!/bin/bash

# --- CONFIGURAZIONE ---
CONTAINER_NAME="andrea_docker-ros1_noetic-1"
NUM_RUNS=30
DURATION=40 

BAG_PREFIX="APF"
DEST_FOLDER="/home/ros_ws/src/dual_pathway_model/simulations/bag_files"
TOPICS="/odom /cmd_vel /odometry/filtered /gazebo/model_states /operator/odom /operator/cmd_vel"

# Path degli script (dentro il container)
SETUP_SCRIPT="/home/ros_ws/src/dual_pathway_model/simulations/random_world_generator.py"


# Goal del ROBOT (Zona C: x=9, y=9)
ROBOT_GOAL_MSG='{
  header: {seq: 0, stamp: {secs: 0, nsecs: 0}, frame_id: "map"},
  pose: {
    position: {x: 9.0, y: 9.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'

TERMINAL_PIDS=()

open_ros_window() {
    local TITLE=$1
    local COMMAND=$2
    gnome-terminal --title="$TITLE" -- bash -c "
        docker exec -it $CONTAINER_NAME bash -c \"
            source /opt/ros/noetic/setup.bash && \
            source /home/ros_ws/devel/setup.bash && \
            $COMMAND
        \"" &   
    TERMINAL_PIDS+=($!)
}

for ((i=1; i<=NUM_RUNS; i++))
do
    echo "--------------------------------------------"
    echo "Inizio Simulazione $i: Seed $i"
    echo "--------------------------------------------"

    # 1. GENERAZIONE MONDO (Eseguito dentro il container)
    # Catturiamo le coordinate dell'operator stampate dallo script
    GOAL_DATA=$(docker exec $CONTAINER_NAME bash -c "python3 $SETUP_SCRIPT $i")
    OP_X=$(echo $GOAL_DATA | cut -d' ' -f1)
    OP_Y=$(echo $GOAL_DATA | cut -d' ' -f2)
    
    echo "[Setup] Operator Goal: X=$OP_X, Y=$OP_Y"

    # 2. Simulation
    open_ros_window "SIM_RUN_$i" "roslaunch apf_planner test_architecture.launch test:=''"
    sleep 15

    # 3. Rosbag recorder
    open_ros_window "RECORDER" "cd $DEST_FOLDER && rosbag record -O ${BAG_PREFIX}_$i.bag $TOPICS"
    sleep 2 

    # 4. Obstacle controller 
    open_ros_window "OBSTACLES_CONTROLLER" "rosrun low_road controller_obstacles.py $OP_X $OP_Y"

    # 5. Starting experiment (publishing robot's goal)
    docker exec $CONTAINER_NAME bash -c "source /opt/ros/noetic/setup.bash && rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped '$ROBOT_GOAL_MSG'"

    sleep $DURATION

    # --- CLEANUP ---
    echo "Fine run $i. Reset in corso..."
    docker exec $CONTAINER_NAME /bin/bash -c "source /opt/ros/noetic/setup.bash && rosnode kill -a"
    sleep 2
    docker exec $CONTAINER_NAME /bin/bash -c "killall -9 rosmaster gzserver"
    pkill -9 -f gzclient
    
    for pid in "${TERMINAL_PIDS[@]}"; do
        kill $pid 2>/dev/null
    done
    TERMINAL_PIDS=()
    sleep 5
done