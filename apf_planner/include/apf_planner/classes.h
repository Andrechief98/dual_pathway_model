#ifndef _CLASSES_H_
#define _CLASSES_H_

#include <vector>
#include <string>
#include <gazebo_msgs/ModelStates.h>

//******* DEFINIZIONE CLASSE GOAL ************
class Goal{
    public:
        int goalID;
        std::vector<double> coordinate={0,0}; //I VETTORI DI VETTORI SONO DA CONSIDERARE TUTTI SU UN'UNICA COLONNA

    //constructor
    Goal(int ID, double x, double y);
};


class Obstacle{
    public:
        std::vector<double> coordinate={0,0};
        double radius;
        int index;
        std::vector<double> F_rep_obs={0,0};

    //constructor
    Obstacle(double x, double y, double r, int i);

    void updateInfo(double x, double y);
};



#endif