# #!/bin/bash

# CONTAINER_NAME="andrea_docker-ros1_noetic-1"
# DURATION=40

# BAG_NAME="APF_dynamic"  
# TOPICS="/odom /cmd_vel /gazebo/model_states /odometry/filtered" 
# DEST_FOLDER="/home/ros_ws/src/dual_pathway_model/bag_files/simulations/dynamic_obstacles"

# MSG='{
#   header: {seq: 0, stamp: {secs: 0, nsecs: 0}, frame_id: "map"},
#   pose: {
#     position: {x: 10.0, y: 0.0, z: 0.0},
#     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
#   }
# }'

# # Function for opening a new window within the container
# open_ros_window() {
#     local TITLE=$1
#     local COMMAND=$2
    
#     gnome-terminal --title="$TITLE" -- bash -c "
#         docker exec -it $CONTAINER_NAME bash -c \"
#             source /opt/ros/noetic/setup.bash && \
#             source /home/ros_ws/devel/setup.bash && \
#             $COMMAND
#         \"; exec bash" & 
# }

# # 1. Simulation
# open_ros_window "SIMULATION" "roslaunch apf_planner test_architecture_dynamic.launch"
# sleep 10

# # 2. Rosbag recorder
# open_ros_window "RECORDER" "cd $DEST_FOLDER && rosbag record -O ${BAG_NAME}.bag $TOPICS"

# # 3. Starting experiment (publishing robot's goal and obstacles controllers)
# open_ros_window "GOAL_PUBLISHER" "rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '$MSG'"
# open_ros_window "OBSTACLES_CONTROLLER" "rosrun low_road controller_obstacles.py"

# sleep $DURATION

# echo "End of the experiement. Closing everything ..."

# # Killing all ROS processes
# docker exec $CONTAINER_NAME /bin/bash -c "source /opt/ros/noetic/setup.bash && rosnode kill -a; killall -9 rosmaster gzserver"

# # Killing Gazebo
# pkill -9 -f gzclient

#!/bin/bash

CONTAINER_NAME="andrea_docker-ros1_noetic-1"
DURATION=40
NUM_RUNS=20


BAG_PREFIX="APF_dynamic"  
TOPICS="/odom /cmd_vel /gazebo/model_states /odometry/filtered" 
DEST_FOLDER="/home/ros_ws/src/dual_pathway_model/bag_files/simulations/dynamic_obstacles"

MSG='{
  header: {seq: 0, stamp: {secs: 0, nsecs: 0}, frame_id: "map"},
  pose: {
    position: {x: 10.0, y: 0.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'

# Array per memorizzare i PID dei terminali aperti
TERMINAL_PIDS=()

# Function for opening a new window within the container
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
  CURRENT_BAG="${BAG_PREFIX}_$i"
  echo "--------------------------------------------"
  echo "Inizio Simulazione $i di $NUM_RUNS: $CURRENT_BAG"
  echo "--------------------------------------------"
    
  # 1. Simulation
  open_ros_window "SIM_RUN_$i" "roslaunch apf_planner test_architecture_dynamic.launch"
  sleep 10

  # 2. Rosbag recorder
  open_ros_window "RECORDER_RUN_$i" "cd $DEST_FOLDER && rosbag record -O ${CURRENT_BAG}.bag $TOPICS"

  # 3. Starting experiment (publishing robot's goal and obstacles controllers)
  open_ros_window "GOAL_PUBLISHER" "rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '$MSG'"
  open_ros_window "OBSTACLES_CONTROLLER" "rosrun low_road controller_obstacles.py"

  sleep $DURATION

  echo "Fine run $i. Pulizia processi..."

  # 4. Kill dei nodi ROS dentro il container
  docker exec $CONTAINER_NAME /bin/bash -c "source /opt/ros/noetic/setup.bash && rosnode kill -a"
  sleep 2
  docker exec $CONTAINER_NAME /bin/bash -c "killall -9 rosmaster gzserver"
  
  # 5. Kill di Gazebo client sull'host
  pkill -9 -f gzclient

  # 6. Chiudiamo solo i terminali aperti in questa run
  for pid in "${TERMINAL_PIDS[@]}"; do
      # Uccidiamo il processo del terminale (e i suoi figli)
      kill $pid 2>/dev/null
  done
  
  # Svuotiamo l'array per la prossima run
  TERMINAL_PIDS=()

  echo "Run $i completata. Attesa ripristino..."
  sleep 5

done
