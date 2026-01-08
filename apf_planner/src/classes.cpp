#include <apf_planner/classes.h>
#include <iostream>
#include <cmath>    //inclusione libreria matematica per eseguire radici quadrate
#include <cstdlib>
#include <apf_planner/functions.h>
#include <gazebo_msgs/ModelStates.h>


#define DIMENSION 2 //dimensione del problema (due dimensioni, x e y)
#define DES_VEL 0.9312//valore di desired velocity (stesso valore sia per Vx che Vy)
#define LAMBDA 0.9928  //valore di lambda del APF (articolo della prof)
#define TIME_STEP 0.2


Goal::Goal(int ID, double x, double y){
        goalID=ID;
        coordinate.clear();

        coordinate.push_back(x);
        coordinate.push_back(y);
    }


Obstacle::Obstacle(double x, double y, double r, int i){
        radius=r;
        index = i;
        coordinate.clear();

        coordinate.push_back(x);
        coordinate.push_back(y);
    }

void Obstacle::updateInfo(double x, double y){
    coordinate.clear();

    coordinate.push_back(x);
    coordinate.push_back(y);
    
}