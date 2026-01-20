#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Odometry.h>
#include <vector>
#include <cmath>
#include <boost/thread.hpp>
#include <iostream>
#include <string>
#include <cstring>
#include <tf2/buffer_core.h>
#include <tf2_ros/transform_listener.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/GetModelState.h>
#include <algorithm>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <apf_planner/apf_planner.h>
#include <apf_planner/functions.h>
#include <apf_planner/classes.h>

PLUGINLIB_EXPORT_CLASS(apf_planner::ApfPlanner, nav_core::BaseLocalPlanner)


//CLASSE LOCAL PLANNER (PLUGIN)
namespace apf_planner{

    ApfPlanner::ApfPlanner() : costmap_ros_(NULL), tf_(NULL), initialized_(false), listener(tfBuffer){}

    ApfPlanner::ApfPlanner(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros) : costmap_ros_(NULL), tf_(NULL), initialized_(false), listener(tfBuffer)
    {
        initialize(name, tf, costmap_ros);
    }

    ApfPlanner::~ApfPlanner() {}


    void ApfPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros)
    {   

        if(!initialized_)
        {   
            nh_ = ros::NodeHandle(name);
            tf_ = tf;

            std::string tf_prefix="";
            nh_.getParam("tf_prefix", tf_prefix); 

            if (tf_prefix==""){
                odom_frame =  "odom";
            }
            else{
                odom_frame = tf_prefix + "/odom";
            }

            if (nh_.getParam("/robot_size/length" , robot_length)) {
                ROS_INFO("APF: '/robot_size/length' parameter found: %f", robot_length);
            } else {
                ROS_WARN("APF: parameter '/robot_size/length' NOT found in %s! Using default value: 1.6", nh_.getNamespace().c_str());
                robot_length = 1.6;
            }

            if (nh_.getParam("/robot_size/width" , robot_width)) {
                ROS_INFO("APF: '/robot_size/width' parameter found: %f", robot_width);
            } else {
                ROS_WARN("APF: parameter '/robot_size/width' NOT found in %s! Using default value: 0.8", nh_.getNamespace().c_str());
                robot_width = 0.8;
            }
            
            sub_odom    =   nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &ApfPlanner::odomCallback, this);
            sub_obs     =   nh_.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, &ApfPlanner::obstacleGazeboCallback, this);
            pub_cmd     =   nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

            initialized_ = true;
            
            goal_reached=false; //ho aggiunto solo questo flag utile per verificare se il goal è stato raggiunto, tutto il resto c'era già
            

        }

        initialized_=true;
        ROS_INFO("APF local planner initialized");
    }

    void ApfPlanner::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        robot_pose_ = *msg;
    }


    void ApfPlanner::obstacleGazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg){

        try{
            if (obstacles_list.size() != msg->name.size()-3){

                // ricreo il vettore degli ostacoli
                obstacles_list.clear();
                for (int i = 0; i < msg->name.size(); i++)
                {
                    std::string name = msg->name[i];
                    // filtra solo gli oggetti che ti interessano
                    if (name != "walls" && name != "ground_plane" && name != "mir")
                    {   
                        
                        // Default radius
                        double radius = 0.3;

                        std::cout << name << std::endl;

                        // Radius of the constraints set according to the considered obstacle:
                        std::string sub_string = "rover";
                        if (name.find(sub_string) != std::string::npos ){
                            radius = 1.0;
                            // std::cout << "############### RADIUS: " << radius << std::endl;
                        }

                        


                        // STARE ATTENTI AL FATTO CHE ros::Time::now() FORNISCE IL TIME DATO DAL CLOCK O DEL PC NEL CASO 
                        // DI ROBOT REALE (usare "use_sim_real == "false"") O DELLA SIMULAZIONE GAZEBO (usare "use_sim_real == "true"")
                        // ros::Time time = ros::Time::now();
                        // Obstacle new_obstacle(msg->pose[i].position.x, msg->pose[i].position.y, radius, i, time);
                        // obstacles_list.push_back(new_obstacle);
                        

                        // CONSIDERING TRASFORMATION:
                        // Costruisci la pose nel frame sorgente (es. "world" o "odom")
                        geometry_msgs::PoseStamped pose_in, pose_out;
                        pose_in.header.frame_id = "odom";  // non è corretto per i modelli gazebo ma l'origin dei modelli gazebo corrisponde a odom
                        pose_in.header.stamp = ros::Time(0);
                        pose_in.pose = msg->pose[i];

                        // Trasforma la pose nel frame "map"
                        try{
                            tf_->transform(pose_in, pose_out, "map");
                        } catch (tf2::TransformException &ex) {
                            ROS_WARN("Transform from odom to map failed: %s", ex.what());
                            continue;
                        }
                        

                        // Estrai coordinate trasformate
                        double x_map = pose_out.pose.position.x;
                        double y_map = pose_out.pose.position.y;

                        // std::cout << "Coordinate ostacolo trasformate: " << x_map << " " << y_map <<std::endl;
                        ros::Time time = ros::Time::now();

                        Obstacle new_obstacle(x_map, y_map, radius, i);
                        obstacles_list.push_back(new_obstacle);
                    }
                }
                N_obs = obstacles_list.size();
                std::cout << "Number of obstacles: " << N_obs << std::endl;

            }
            else{
        
                // se l'obstacle_list non è vuoto allora si devono aggiornare le posizioni degli oggetti
                // e le corrispondenti velocità
                if (!obstacles_list.empty()) {
                    for (int i = 0; i < obstacles_list.size(); i++) {
                        int index = obstacles_list[i].index;

                        if (index >= msg->pose.size()) {
                            ROS_WARN_THROTTLE(1.0, "Obstacle index %d out of range", index);
                            continue;
                        }

                        // --- Costruisci la pose nel frame odom (Gazebo usa tipicamente odom come riferimento)
                        geometry_msgs::PoseStamped pose_in, pose_out;
                        pose_in.header.frame_id = "odom";
                        pose_in.header.stamp = ros::Time(0);
                        pose_in.pose = msg->pose[index];

                        try {
                            // --- Trasforma la posizione nel frame "map"
                            tf_->transform(pose_in, pose_out, "map");
                        } catch (tf2::TransformException &ex) {
                            ROS_WARN_THROTTLE(1.0, "Transform from odom to map failed: %s", ex.what());
                            continue;  // passa al prossimo ostacolo
                        }

                        // --- Estrai coordinate trasformate
                        double x_map = pose_out.pose.position.x;
                        double y_map = pose_out.pose.position.y;
                        ros::Time time = ros::Time::now();

                        // --- Aggiorna la posizione dell’ostacolo nel frame map
                        obstacles_list[i].updateInfo(x_map, y_map);

                        // DEBUG opzionale
                        // ROS_INFO_STREAM("Obstacle radius:" << obstacles_list[i].r <<"");
                        // ROS_INFO_STREAM("Updated obstacle[" << i << "] in map frame: ("
                        //                 << x_map << ", " << y_map << ")");
                        
                    }
                }

            }
            // std::cout << "Number of obstacles detected: " << N_obs << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }

    }

    void ApfPlanner::set_Position_Orientation_Info(){
        /*
            dal messaggio dell'odometria (che viene salvato in una variabile globale - robot_pose_ - tramite la funzione corrispondente di 
            callback) vado a salvare le coordinate attuali del robot e la sua orientazione attuale (questa sarà espressa in quaternioni)
        */

        curr_robot_coordinates={robot_pose_.pose.pose.position.x, robot_pose_.pose.pose.position.y}; 
        curr_robot_orientation=tf2::getYaw(robot_pose_.pose.pose.orientation); 

        std::cout << "\n";
        std::cout << "**VETTORI COORDINATE GOAL E COORDINATE ROBOT CALCOLATI: " << std::endl;
        std::cout << "  *Goal coordinates: " << goal_coordinates[0] << " " << goal_coordinates[1] << std::endl;
        std::cout << "  *Goal orientation z-axis (radians) [-pi,pi]: " << goal_orientation << std::endl;
        std::cout << "\n";
        std::cout << "  *Robot coordinates: " << curr_robot_coordinates[0] << " " << curr_robot_coordinates[1] << std::endl;
        std::cout << "  *Robot orientation z-axis (radians) [-pi,pi]: " << curr_robot_orientation << std::endl; 

        return;
    }

    void ApfPlanner::setVelocityInfo(){
        /*
            Attraverso questa funzione estraggo semplicemente le informazioni contenute nel messaggio dell'odometria ma legate alla velocità 
            lineare e angolare. Una volta estratte però è necessario riferire tali velocità rispetto al frame "map" per poter poi fare in seguito
            i calcoli delle forze.
        */

        curr_robot_lin_vel={robot_pose_.twist.twist.linear.x, robot_pose_.twist.twist.linear.y}; 
        curr_robot_ang_vel=robot_pose_.twist.twist.angular.z;
        
        std::cout << "\n";
        std::cout << "**VETTORE VELOCITÀ ROBOT ATTUALE RISPETTO AL BASE_FOOTPRINT (frame del robot): " << std::endl;
        std::cout << "  *Robot velocity vector  : " << curr_robot_lin_vel[0] << " " << curr_robot_lin_vel[1] << std::endl; 

        //ESECUZIONE TRASFORMATA:
        //ci serve trasformare la velocità dal base_footprint al frame map

        ros::Rate rate(10.0);
        geometry_msgs::TransformStamped tf_result; //creo la variabile che contiene la trasformata (traslazione+rotazione)
        try {
            //salvo in "tf_result" la trasformata che mi porta dal frame "base_footprint" --> "map"
            tf_result = tfBuffer.lookupTransform("map", "base_footprint", ros::Time(0)); 
        } 
        catch (tf2::TransformException& ex) {
            //NON HO MESSO NULLA DA FARE NEL CASO IN CUI LA TRASFORMATA NON FUNZIONI
            ROS_ERROR("[APF LOCAL PLANNER]: transform from base_footprint to map failed");
        }

        /*creo un quaternione che rappresenta la rotazione
         necessaria per passare da un frame all'altro. Tale quaternione deve essere uguale a 
         quello dato dalla trasformata relativa ai frame che stiamo considerando*/
        tf2::Quaternion q(
            tf_result.transform.rotation.x,
            tf_result.transform.rotation.y,
            tf_result.transform.rotation.z,
            tf_result.transform.rotation.w
        );

        /*creo un vettore (con dimensione 3) che rappresenta la traslazione
         necessaria per passare da un frame all'altro. In tal caso, non dovendo
         eseguire nessuna traslazione (visto che dobbiamo eseguire la trasformata 
         di una velocità), andiamo a creare una traslazione nulla*/
        tf2::Vector3 p(0,0,0);

        //creo un vettore trasformata
        tf2::Transform transform(q, p);

        //converto i vettori velocità da 2D a 3D (lungo z metto semplicemente 0)
        tf2::Vector3 velocity_in_child_frame(curr_robot_lin_vel[0],curr_robot_lin_vel[1],0);

        //eseguo la trasformata (usando il metodo precedente le nuove coordinate saranno ottenute semplicemente tramite moltiplicazione)
        tf2::Vector3 velocity_in_target_frame = transform * velocity_in_child_frame;

        //aggiorno il vettore velocità attuale del robot con i nuovi valori rispetto al frame "map"
        curr_robot_lin_vel={velocity_in_target_frame[0],velocity_in_target_frame[1]};

        //calcolo la direzione attuale del robot rispetto al frame "map" come versore della velocità attuale:
        for(int i=0; i<n_robot.size();i++){
            n_robot[i]=curr_robot_lin_vel[i]/vect_norm1(curr_robot_lin_vel);
        }

        //stampo informazioni
        std::cout << "VELOCITY NEL BASE LINK FRAME: " << std::endl;
        std::cout<< velocity_in_target_frame[0] << " " << velocity_in_target_frame[1]<< std::endl;
        std::cout<< curr_robot_lin_vel[0] << " " << curr_robot_lin_vel[1] << std::endl;


        return;
    }


    void ApfPlanner::computeAttractiveForce(){
        /*
            Calcolo della forza attrattiva secondo l' APF
        */
       
        F_att={0,0};
        e=compute_direction(goal_coordinates,curr_robot_coordinates);

        for(int i=0; i<e.size(); i++){
            //calcolo forza attrattiva
            F_att[i]=(desired_vel*e[i]-curr_robot_lin_vel[i])/alfa;
        }
        std::cout << "vettore e:"<< std::endl;
        std::cout << e[0] << " " << e[1] << std::endl;
        std::cout << "forza attrattiva goal calcolata" << std::endl;
        std::cout << F_att[0] << " " << F_att[1] << std::endl;

        return;
    }



    void ApfPlanner::computeObstacleRepulsiveForce(std::vector<Obstacle> obs_list){
        
        F_rep_obs_tot = {0,0};

        for(int i = 0; i<obs_list.size(); i++){

            //CALCOLO EFFETTIVO DELLA FORZA REPULSIVA PER OGNI OSTACOLO
            double F_fov=0;
            std::vector<double> n_obs={0,0};

            Obstacle obstacle = obs_list[i];

            n_obs=compute_direction(curr_robot_coordinates, obstacle.coordinate);  
            F_fov=lambda+(1-lambda)*((1+compute_cos_gamma(n_robot, n_obs))/2); 


            // 3. Calcolo dell'angolo relativo tra l'asse del robot e l'ostacolo
            double angle_to_obs = atan2(obstacle.coordinate[1] - curr_robot_coordinates[1], 
                                    obstacle.coordinate[0] - curr_robot_coordinates[0]);
            double alpha = angle_to_obs - curr_robot_orientation; // Angolo relativo al corpo robot

            // 4. Calcolo del raggio locale dell'ellisse (R_ell)
            double cos_a = cos(alpha);
            double sin_a = sin(alpha);

            double a = robot_length/2;
            double b = robot_width/2;

            double r_robot_local = (a * b) / sqrt(pow(b * cos_a, 2) + pow(a * sin_a, 2));

            double distance = vect_norm2(curr_robot_coordinates, obstacle.coordinate) - obstacle.radius ; //- r_robot_local;
            std::cout << "Obstacle distance: " << distance << std::endl;
            for(int k=0; k < obstacle.F_rep_obs.size(); k++){

                // First possible equation:
                obstacle.F_rep_obs[k] = exp(3-(distance/obstacle.radius))*F_fov*n_obs[k];

                // Second possible equation:
                //obstacle.F_rep_obs[k] = A*exp(-distance/B)*F_fov*n_obs[k];


                F_rep_obs_tot[k] = F_rep_obs_tot[k] + obstacle.F_rep_obs[k];
            }
            
            std::cout << "Obstacle repulsive force computed" << std::endl;
            std::cout << F_rep_obs_tot[0] << " " << F_rep_obs_tot[1] << std::endl;

            
            }
        
        std::cout << "Total obstacle repulsive force computed" << std::endl;
        std::cout << F_rep_obs_tot[0] << " " << F_rep_obs_tot[1] << std::endl;

    }


    void ApfPlanner::computeTotalForce(){

        F_tot={0,0};
        for(int i=0; i<F_tot.size(); i++){

            F_tot[i]=F_att[i]+F_rep_obs_tot[i];   

            if(std::fabs(F_tot[i])>max_lin_acc_x){
                F_tot[i]=sign(F_tot[i])*max_lin_acc_x;
            }
        }

        std::cout << "\n";
        std::cout << "**TOTAL FORCE COMPUTED: " << std::endl;
        std::cout << "  *Total force vector  : " << F_tot[0] << " " << F_tot[1] << std::endl; 
    }


    bool ApfPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan)
    {   

        if(!initialized_)
        {
            ROS_ERROR("This planner has not been initialized");
            return false;
        }

        /*Istruzioni già presenti di default nel plugin. Semplicemente viene salvato all'interno di tale variabile globale
         "global_plan_" (vettore di geometry_msgs::PoseStamped già presente di default nel plugin) le coordinate dei vari 
          punti corrispondenti al GLOBAL PATH. A NOI INTERESSA COMUNQUE L'ULTIMO ELEMENTO DEL VETTORE, IL QUALE CONTERRÀ LE 
          INFORMAZIONI DEL GOAL*/
        global_plan_.clear();
        global_plan_ = orig_global_plan;
        
        //quando settiamo un nuovo goal (planner frequency 0 Hz nel config file .yaml -> global planner chiamato una volta, solo all'inizio), 
        //resettiamo il flag. In questo modo in seguito potremo eseguire delle verifiche per capire se il goal è stato raggiunto
        goal_reached=false;

        //puliamo anche il vettore di coordinate che conteneva le coordinate del goal precedente
        goal_coordinates.clear();

        //Salviamo quindi solo l'ultimo punto del vettore che contiene tutto il global path
        int size_global_plan=global_plan_.size();
        goal_pose_=global_plan_[size_global_plan-1];

        //Setto le coordinate del goal all'interno di variabili globali che adopererò poi nel resto del programma:
        goal_coordinates={goal_pose_.pose.position.x, goal_pose_.pose.position.y}; // corretto
        goal_orientation=tf2::getYaw(goal_pose_.pose.orientation); //corretto
        
        std::cout << "COORDINATE GOAL RICEVUTE: " << std::endl;
        std::cout << "Pose Frame : " << goal_pose_.header.frame_id << std::endl; //FRAME = "map" (coincide con /odom)
        std::cout << "  Coordinates (meters) : " << goal_coordinates[0] << " " << goal_coordinates[1] << std::endl;
        std::cout << "  Orientation z-axis (radians) : " << goal_orientation << std::endl;

        return true;
    }


    bool ApfPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
    {

        if(!initialized_)
        {
            ROS_ERROR("This planner has not been initialized");
            return false;
        }

        //Adopero semplicemente le funzioni create in precedenza
        set_Position_Orientation_Info();
        setVelocityInfo();

        computeAttractiveForce();
        computeObstacleRepulsiveForce(obstacles_list);
        computeTotalForce();

        /*A questo punto, una volta ottenuta la forza totale, posso procedere a determinare quale sarà la nuova linear velocity*/

        //CALCOLO NUOVA VELOCITA'
        new_robot_lin_vel={0,0};
        new_robot_pos={0,0};
        

        //VERIFICA RAGGIUNGIMENTO GOAL 
        if(vect_norm2(goal_coordinates,curr_robot_coordinates)<=distance_tolerance){
            std::cout << " ------------- Distanza raggiunta ----------------" << std::endl;
            //coordinate raggiunte. Vettore velocità lineare rimane nullo.
            new_robot_lin_vel={0,0};
            

           
            if(std::fabs(angles::shortest_angular_distance(curr_robot_orientation,goal_orientation))<=angle_tolerance){
                //anche l'orientazione del goal è stata raggiunta
                new_robot_ang_vel_z=0;
                goal_reached=true;
                std::cout<<"Orientazione goal raggiunta"<<std::endl;
                std::cout<<"GOAL RAGGIUNTO"<<std::endl;
            }
            else{
                // Se le coordinate del goal sono state raggiunte ma l'orientazione finale no, la velocità angolare deve 
                // essere calcolata per far ruotare il robot nella posa finale indicata
                std::cout << "Orientazione non raggiunta" << std::endl;
                new_robot_ang_vel_z=K_p*(angles::shortest_angular_distance(curr_robot_orientation,goal_orientation));
                }
            
        }
        else{
            std::cout << "------------- Distanza non raggiunta ----------------" << std::endl;

            for(int i=0; i<new_robot_lin_vel.size(); i++){
                new_robot_lin_vel[i]=curr_robot_lin_vel[i]+delta_t*F_tot[i];// GROSSO PROBLEMA (SE F_tot È NEGATIVA ALLORA LA VELOCITÀ TENDE A RIDURSI A LIVELLO GLOBALE)
                
                if(std::fabs(new_robot_lin_vel[i])>desired_vel){
                
                    new_robot_lin_vel[i]=sign(new_robot_lin_vel[i])*desired_vel;
                }

                new_robot_pos[i]=curr_robot_coordinates[i]+delta_t*new_robot_lin_vel[i]; //adoperiamo la velocità del modello (calcolata dalle forze) anzichè quella effettiva del robot
            }

            beta=std::atan2(new_robot_pos[1]-curr_robot_coordinates[1],new_robot_pos[0]-curr_robot_coordinates[0]);

            std::cout << "-------- INFO PER ANGOLI -------" << std::endl;
            std::cout << "  *beta= " << beta << std::endl;
            std::cout << "  *Robot orientation z-axis (radians) [-pi,pi] : " << curr_robot_orientation << std::endl; 
            std::cout << "  *Goal orientation z-axis (radians) [-pi,pi] : " << goal_orientation << std::endl;
            std::cout << "\n";
            std::cout << "-------- NUOVA VELOCITÀ ROBOT CALCOLATA --------- " << std::endl;
            std::cout << "  *New position : " << new_robot_pos[0] << " " << new_robot_pos[1] << std::endl;
            std::cout << "  *New velocity vector  : " << new_robot_lin_vel[0] << " " << new_robot_lin_vel[1] << std::endl; 
            std::cout << "  *New angular velocity : " << new_robot_ang_vel_z << std::endl;

            if(std::fabs(angles::shortest_angular_distance(curr_robot_orientation,beta))<=_PI/2){
                //la rotazione per muovere il robot nella direzione della forza è compresa in [-pi/2;pi/2]
                //possiamo eseguire una combo di rotazione e movimento in avanti

                //ROTAZIONE:
                new_robot_ang_vel_z=K_p*(angles::shortest_angular_distance(curr_robot_orientation, beta));

                if (std::fabs(new_robot_ang_vel_z)>max_angular_vel_z){
                    new_robot_ang_vel_z=sign(new_robot_ang_vel_z)*max_angular_vel_z;
                }

                //TRASLAZIONE GIA' CALCOLATA (RISPETTO AL FRAME "map")

            }

            else{
                //è preferibile far ruotare il robot verso la direzione della futura posizione prima di farlo muovere linearmente
                
                new_robot_lin_vel={0,0};
                new_robot_ang_vel_z=sign(angles::shortest_angular_distance(curr_robot_orientation,beta))*max_angular_vel_z;

            }

            // // new_robot_ang_vel_z=K_p*(angles::shortest_angular_distance(curr_robot_orientation, std::atan2(goal_coordinates[1]-curr_robot_coordinates[1],goal_coordinates[0]-curr_robot_coordinates[0])));
            
        }

        //NECESSARIO TRASFORMARE LA VELOCITÀ DA "map" AL FRAME DEL ROBOT "/base_link")
        ros::Rate rate(10.0);
        geometry_msgs::TransformStamped tf_result;
        try {
            tf_result = tfBuffer.lookupTransform("base_link", "map", ros::Time(0));
        } 
        catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "TF lookupTransform failed (map -> base_link): %s", ex.what());
            cmd_vel = geometry_msgs::Twist();
            return false;
        }

        
        tf2::Quaternion q(
            tf_result.transform.rotation.x,
            tf_result.transform.rotation.y,
            tf_result.transform.rotation.z,
            tf_result.transform.rotation.w
        );
        tf2::Vector3 p(0,0,0);

        tf2::Transform transform(q, p);
        tf2::Vector3 velocity_in_child_frame(new_robot_lin_vel[0],new_robot_lin_vel[1],0);
        tf2::Vector3 velocity_in_target_frame = transform * velocity_in_child_frame;

        std::cout << "VELOCITY NEL BASE LINK FRAME: " << std::endl;
        std::cout<< velocity_in_target_frame[0] << " " << velocity_in_target_frame[1]<< std::endl;


        //PUBBLICAZIONE MESSAGGIO
        pub_cmd = nh_.advertise<geometry_msgs::Twist>("/cmd_vel",1);

        cmd_vel.angular.x=0.0;
        cmd_vel.angular.y=0.0;
        cmd_vel.angular.z=new_robot_ang_vel_z;
        // cmd_vel.linear.x=vect_norm1(new_robot_lin_vel);
        // cmd_vel.linear.y=0.0;
        cmd_vel.linear.x=std::fabs(velocity_in_target_frame[0]);
        cmd_vel.linear.y=0.0;
        cmd_vel.linear.z=0.0;

        pub_cmd.publish(cmd_vel);

        std::cout << "\nmessaggio pubblicato\n" << std::endl;
        std::cout << "\n-------------------------------------------------------------------\n\n\n" << std::endl;

        return true;
    }

    bool ApfPlanner::isGoalReached()
    {   
        /*
            Funzione già presente di default nel plugin.
            Viene eseguita prima di eseguire "computeVelocityCommands() per capire se il goal è stato raggiunto"
        */
        if(!initialized_)
        {
            ROS_ERROR("This planner has not been initialized");
            return false;
        }
        if(goal_reached){
            return true;
        }
        else{
            return false;
        }
        
    }
}
