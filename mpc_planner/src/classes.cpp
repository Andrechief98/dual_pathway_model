#include <mpc_planner/classes.h>
#include <iostream>
#include <cmath>    //inclusione libreria matematica per eseguire radici quadrate
#include <cstdlib>
#include <gazebo_msgs/ModelStates.h>

Obstacle::Obstacle(double x, double y, double radius, int i, ros::Time current_time){
        r=radius;
        index = i;
        pos(0) = x;
        pos(1) = y;
        previous_time = current_time;
    }

void Obstacle::updateInfo(double x, double y, ros::Time current_time){
    Eigen::Vector2d new_pos{x, y};
    delta_t = std::max(1e-4,(current_time - previous_time).toSec());
    // std::cout << "delta_t: " << (current_time - previous_time).toSec() << std::endl;
    vel = (new_pos-pos)/delta_t; 
    pos = new_pos;
    previous_time = current_time;
    
}



