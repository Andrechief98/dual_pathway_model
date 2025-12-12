FROM osrf/ros:noetic-desktop-full

# Sourcing ros
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Installing apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-rosdep \
    python3-pip \
    curl \
    nano \
    ros-noetic-navigation \
    ros-noetic-mir-robot \
    ros-noetic-realsense2-camera \
    ros-noetic-realsense2-description \
    python3-tk \
    build-essential \
    liblapack-dev \
    pkg-config \
    gfortran \
    coinor-libipopt-dev \
    wget

    RUN pip3 install ollama

# Installing casadi
WORKDIR /opt/casadi
RUN git clone https://github.com/casadi/casadi.git .
RUN mkdir build && cd build && cmake .. -DWITH_BUILD_REQUIRED=ON -DWITH_IPOPT=ON -DCMAKE_INSTALL_PREFIX=/usr/local
RUN cd build && make install

# Cloning project
WORKDIR /home/ros_ws/src 
RUN git clone https://github.com/Andrechief98/dual_pathway_model.git

# Installing project's ros dependencies
RUN rosdep init || true
RUN rosdep update

RUN rosdep install --from-paths . --ignore-src -r -y

# Installing SFM plugin for actor in Gazebo simulations
RUN git clone https://github.com/robotics-upo/lightsfm.git
RUN git clone https://github.com/robotics-upo/gazebo_sfm_plugin.git
WORKDIR /home/ros_ws/src/lightsfm
RUN /bin/bash -c "make && sudo make install"

WORKDIR /home/ros_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Sourcing the ros workspace
RUN echo "source /home/ros_ws/devel/setup.bash" >> /root/.bashrc

# Disable ROS Noetic EOL
RUN echo "export DISABLE_ROS1_EOL_WARNINGS="true"" >> /root/.bashrc

# Installing ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /home/ros_ws
RUN echo "ALL DONE"

