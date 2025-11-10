#ifndef _CLASSES_H_
#define _CLASSES_H_

#include <vector>
#include <string>
#include <gazebo_msgs/ModelStates.h>
#include <Eigen/Dense>



class Obstacle{
    public:
        Eigen::Vector2d pos{0, 0};
        Eigen::Vector2d vel{0, 0};
        Eigen::Vector3d bounding_box{0,0,0};
        double r = 0;
        int index;
        double delta_t=0;
        ros::Time previous_time;

    //constructor
    Obstacle(double x, double y, double radius, int i, ros::Time current_time);

    
    void updateInfo(double x, double y, ros::Time current_time);
};



#endif