#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>
#include <std_srvs/Empty.h>
#include <nav_msgs/Path.h>
#include <vector>
#include <cmath>
#include <boost/thread.hpp>
#include <iostream>
#include <string>
#include <cstring>
#include <angles/angles.h>
#include <tf2/buffer_core.h>
#include <tf2_ros/transform_listener.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/GetModelState.h>
#include <sensor_msgs/LaserScan.h>
#include <algorithm>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <mpc_planner/mpc_planner.h>
#include <casadi/casadi.hpp>
#include <yaml-cpp/yaml.h>
#include <gazebo_msgs/ModelStates.h>
#include <mpc_planner/classes.h>
#include "mpc_planner/mpcParameters.h"




PLUGINLIB_EXPORT_CLASS(mpc_planner::MpcPlanner, nav_core::BaseLocalPlanner)


namespace mpc_planner {

    namespace cs = casadi;

    MpcPlanner::MpcPlanner() : costmap_ros_(NULL), tf_(NULL), initialized_(false), listener(tfBuffer){}

    MpcPlanner::MpcPlanner(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros) : costmap_ros_(NULL), tf_(NULL), initialized_(false), listener(tfBuffer)
    {
        initialize(name, tf, costmap_ros);
    }

    MpcPlanner::~MpcPlanner() {}

    void MpcPlanner::buildSolver(){

        // Stato: X ∈ ℝ^{nx × (Np+1)}
            // [x0,y0,th0,...,xN,yN,thN]
        cs::MX X = cs::MX::sym("X", nx*(Np+1));

        // Input: U ∈ ℝ^{nu × Np}
            // [v0, w0, v1, w1, ..., vN-1, wN-1]
        cs::MX U = cs::MX::sym("U", nu*Np);

        // Slack variables per vincoli terminali (sulla distanza dal punto finale della reference)
            // s ∈ ℝ^{ns × 1}
        cs::MX s = cs::MX::sym("s", ns);

        // Slack variables per avoidance ostacoli
        // Ogni ostacolo ha un solo slack per step (adatta se serve di più)
        int ns_obs = Np * N_obs;
        cs::MX s_obs = cs::MX::sym("s_obs", ns_obs);


        // parameters of the optimization problem: 
        //      [x0,y0,th0, reference(3*(N+1)), Qx, Qy, Qth, Rv, Rw, Px, Py, Pth, alfa[obs1, obs2, ...], beta[obs1, obs2, ...], last_v, last_w, obs_info([x_obs1, y_obs1, vx_obs1, vy_obs1, r_obs1], [x_obs2, y_obs2, vx_obs2, vy_obs2, r_obs2]), ...]   
        int p_dim = nx                       // stato iniziale
          + nx * (Np + 1)                   // traiettoria di riferimento
          + N_cost_params                   // parametri di costo
          + nu                              // ultimo comando (warm start)
          + N_obs * N_obs_info;             // info sugli ostacoli
        
        // p ∈ ℝ^{p_dim × 1}
        cs::MX p = cs::MX::sym("p", p_dim, 1);

        // Helper: indice del blocco pesi dentro p
        int weights_start_idx = nx + nx*(Np+1);

        // Extraction of the weights from p
        cs::MX Qx   =   p(weights_start_idx + 0);
        cs::MX Qy   =   p(weights_start_idx + 1);
        cs::MX Qth  =   p(weights_start_idx + 2);
        cs::MX Rv   =   p(weights_start_idx + 3);
        cs::MX Rw   =   p(weights_start_idx + 4);
        cs::MX Px   =   p(weights_start_idx + 5);
        cs::MX Py   =   p(weights_start_idx + 6);
        cs::MX Pth  =   p(weights_start_idx + 7);

        cs::MX alfa_vec =   p(cs::Slice(weights_start_idx + 8, weights_start_idx + 8 + N_obs));
        cs::MX beta_vec =   p(cs::Slice(weights_start_idx + 8 + N_obs, weights_start_idx + 8 + 2 * N_obs));          
        
        // L'offset per i parametri successivi (last_u e info ostacoli)
        int last_u_idx = weights_start_idx + 8 + (2 * N_obs);
        int obstacle_info_start = last_u_idx + nu;

        // Cost function
        cs::MX J = cs::MX::zeros(1);

        for (int k = 0; k < Np; ++k) {

            // Stato corrente
            cs::MX xk = X(cs::Slice(nx*k, nx*(k+1))); // 1 × nx

            cs::MX x_r = p(cs::Slice(nx + nx*k, nx + nx*(k+1))); // 1 × nx

            
            cs::MX vk = U(2*k + 0);
            cs::MX wk = U(2*k + 1);

            // Differenza angolo wrapped
            cs::MX raw = x_r(2) - xk(2);
            // cs::MX diff_th = cs::MX::atan2(cs::MX::sin(raw), cs::MX::cos(raw));
            cs::MX diff_th = 1.0 - cs::MX::cos(x_r(2) - xk(2));

            // Costo tracking
            J = J
                + Qx * (x_r(0) - xk(0)) * (x_r(0) - xk(0))
                + Qy * (x_r(1) - xk(1)) * (x_r(1) - xk(1))
                + Qth * diff_th //* diff_th
                + Rv * vk * vk
                + Rw * wk * wk;

            // Obstacle avoidance
            for(int j=0; j<N_obs; j++){

                cs::MX alfa_j = alfa_vec(j);
                cs::MX beta_j = beta_vec(j);
                int obs_ptr = obstacle_info_start + N_obs_info * j;

                cs::MX obs_pos = p(cs::Slice(obs_ptr, obs_ptr + 2));
                cs::MX obs_vel = p(cs::Slice(obs_ptr + 2, obs_ptr + 4));
                cs::MX obs_r   = p(obs_ptr + 4);


                // Posizione futura ostacolo
                cs::MX fut_obs_pos = obs_pos + k * dt * obs_vel;

                cs::MX diff = xk(cs::Slice(0,2)) - fut_obs_pos;
                cs::MX distance = cs::MX::sqrt(cs::MX::sum1(diff*diff))- obs_r;

                distance = cs::MX::fmax(distance, 0.001);

                // Penalty logaritmico
                cs::MX obstacle_penalty = -alfa_j * cs::MX::log(beta_j * distance); //alfa/(0.05*(distance * distance))

                // Aggiunta costo ostacolo
                J = J + obstacle_penalty;
            }
        }

        // Terminal cost
        // Stato terminale
        cs::MX xN = X(cs::Slice(nx*Np, nx*(Np+1)));// 1 × nx

        // Riferimento terminale
        cs::MX x_rN = p(cs::Slice(nx + nx*Np, nx + nx*(Np+1))); // 1 × nx

        // Differenza angolare "wrapped"
        cs::MX raw_N = x_rN(2) - xN(2);
        //cs::MX diff_th_N = cs::MX::atan2(cs::MX::sin(raw_N), cs::MX::cos(raw_N));
        cs::MX diff_th_N = 1.0 - cs::MX::cos(x_rN(2) - xN(2)); 

        double terminal_slack_penalty = 30.0;

        // Terminal tracking cost + slack cost
        J = J
            + Px * (x_rN(0) - xN(0)) * (x_rN(0) - xN(0))
            + Py * (x_rN(1) - xN(1)) * (x_rN(1) - xN(1))
            + Pth * diff_th_N //* diff_th_N
            + terminal_slack_penalty * s * s;

        // Cost for obstacle slack variables
        double slack_penalty = 1e5;

        // Somma dei quadrati di tutti gli s_obs
        for (int idx = 0; idx < ns_obs; ++idx) {
            cs::MX s_i = s_obs(idx);
            J = J + slack_penalty * s_i * s_i;
        }


        
    
        // CONSTRAINTS DEFINITION 
        //  - initial condition
        //  - dynamics (Euler / RK4)
        //  - slack variables for obstacles (soft constraints, ellipsoid)

        std::vector<cs::MX> g;    // lista dei vincoli MX

        // Stato iniziale vincolato al robot (primi nx elementi di p)
        cs::MX x_init = p(cs::Slice(0, nx));        // 1 x nx
        cs::MX x0 = X(cs::Slice(0, nx));            // 1 x nx

        // Vincolo iniziale: X0 == x_init
        g.push_back(x0 - x_init);
        lbg = cs::DM::zeros(nx);
        ubg = cs::DM::zeros(nx);


        // Parametri robot per ellisse (usati simbolicamente solo come numerici qui)
        double a_robot = robot_length / 2.0; // semiasse lungo
        double b_robot = robot_width  / 2.0; // semiasse corto

        // LOOP ORIZZONTE
        for (int k = 0; k < Np; ++k) {

            // Stati e comandi (2D indexing)
            cs::MX xk  = X(cs::Slice(nx*k, nx*(k+1)));
            cs::MX xk1 = X(cs::Slice(nx*(k+1), nx*(k+2)));
            cs::MX vk  = U(nu*k + 0);
            cs::MX wk  = U(nu*k + 1);

            // ===== DYNAMIC CONSTRAINT =====
            cs::MX dyn;
            if (model == "euler") {
                cs::MX x_next = xk(0) + dt * vk * cs::MX::cos(xk(2));
                cs::MX y_next = xk(1) + dt * vk * cs::MX::sin(xk(2));
                cs::MX th_next = xk(2) + dt * wk;
                // th_next = cs::MX::atan2(cs::MX::sin(th_next), cs::MX::cos(th_next));

                dyn = cs::MX::vertcat({
                    xk1(0) - x_next,
                    xk1(1) - y_next,
                    xk1(2) - th_next
                });
            }
            else if (model == "RK4") {
                // k1
                cs::MX dx1 = vk * cs::MX::cos(xk(2));
                cs::MX dy1 = vk * cs::MX::sin(xk(2));
                cs::MX dth1 = wk;
                cs::MX k1 = cs::MX::vertcat({dx1, dy1, dth1});

                // k2
                cs::MX x_temp = xk + (dt/2.0) * k1;
                cs::MX dx2 = vk * cs::MX::cos(x_temp(2));
                cs::MX dy2 = vk * cs::MX::sin(x_temp(2));
                cs::MX dth2 = wk;
                cs::MX k2 = cs::MX::vertcat({dx2, dy2, dth2});

                // k3
                x_temp = xk + (dt/2.0) * k2;
                cs::MX dx3 = vk * cs::MX::cos(x_temp(2));
                cs::MX dy3 = vk * cs::MX::sin(x_temp(2));
                cs::MX dth3 = wk;
                cs::MX k3 = cs::MX::vertcat({dx3, dy3, dth3});

                // k4
                x_temp = xk + dt * k3;
                cs::MX dx4 = vk * cs::MX::cos(x_temp(2));
                cs::MX dy4 = vk * cs::MX::sin(x_temp(2));
                cs::MX dth4 = wk;
                cs::MX k4 = cs::MX::vertcat({dx4, dy4, dth4});

                cs::MX x_next = xk + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
                dyn = xk1 - x_next;
            }
            else {
                throw std::runtime_error("Model type not recognized (must be 'euler' or 'RK4')");
            }

            // aggiungi vincolo dinamico
            g.push_back(dyn);
            lbg = cs::DM::vertcat({lbg, cs::DM::zeros(nx)});
            ubg = cs::DM::vertcat({ubg, cs::DM::zeros(nx)});    

            // OBSTACLE AVOIDANCE (soft ellipsoidal constraints)
            // Per coerenza con lo stato che è x_{k+1}, usiamo xk1 e fut_obstacle corrispondente
            for (int j = 0; j < N_obs; ++j) {

                // Indice base nel vettore p per l'ostacolo j
                int obs_base = obstacle_info_start + j * N_obs_info;

                // Estrazione posizione e velocità (assumo ordine: x, y, vx, vy, r)
                cs::MX obs_pos = p(cs::Slice(obs_base, obs_base + 2));      
                cs::MX obs_vel = p(cs::Slice(obs_base + 2, obs_base + 4));  
                cs::MX obs_r   = p(obs_base + 4);                           

                // Posizione futura dell'ostacolo (uso (k+1)*dt dato che constraint su xk1)
                cs::MX fut_obs_pos = obs_pos + (k+1) * dt * obs_vel; // 2x1

                // Relativa (xk1 - fut_obs_pos)
                cs::MX dx = xk1(0) - fut_obs_pos(0);
                cs::MX dy = xk1(1) - fut_obs_pos(1);

                // Rotazione nel frame del robot 
                cs::MX cos_m = cs::MX::cos(-xk1(2));
                cs::MX sin_m = cs::MX::sin(-xk1(2));
                cs::MX dxr = cos_m * dx - sin_m * dy;
                cs::MX dyr = sin_m * dx + cos_m * dy;

                // Inflazione ellisse
                double safety_margin = 1.1;
                cs::MX a_infl = a_robot + safety_margin*obs_r;
                cs::MX b_infl = b_robot + safety_margin*obs_r;

                // Vincolo ellittico: 1- (dxr^2 / a_infl^2) + (dyr^2 / b_infl^2) <= 0 
                cs::MX obs_constr = 1 - (dxr*dxr) / (a_infl*a_infl) - (dyr*dyr) / (b_infl*b_infl);

                // Slack corrispondente 
                int slack_index = k*N_obs + j; 
                cs::MX s_obs_kj = s_obs(slack_index);

                // Soft constraint: obs_constr <= s_obs_jk  <=>  obs_constr - s <= 0
                cs::MX g_obs = obs_constr - s_obs_kj;

                g.push_back(g_obs);
                lbg = cs::DM::vertcat({lbg, cs::DM::ones(1)*-1e20});     // lower bound 
                ubg = cs::DM::vertcat({ubg, cs::DM::zeros(1)});          // upper bound 
            }
        }



        // ===== Terminal constraint (soft, slack s) =====
        cs::MX terminal_constr = cs::MX::sumsqr(x_rN(cs::Slice(0,2)) - xN(cs::Slice(0,2))) - cs::MX::sum1(cs::MX::pow(s,2));
        g.push_back(terminal_constr);
        lbg = cs::DM::vertcat({lbg, cs::DM::ones(ns)*-1e-3});       // lb < x_rN - xN - s
        ubg = cs::DM::vertcat({ubg, cs::DM::zeros(ns)});            // x_rN - xN -s < ub

        // ===== Optimization variables bounds =====
        int nX = nx*(Np+1);
        int nU = nu*Np;
        int n_opt = nX + nU; // inizialmente solo X+U

        // Optimization variables constraints
        lbx_full = cs::DM::ones(n_opt) * -1e20;
        ubx_full = cs::DM::ones(n_opt) *  1e20;

        // State constraints:
        for (int i = 0; i < nX; ++i) {
            lbx_full(i) = -1e20;
            ubx_full(i) =  1e20;
        }

        // Input constraints:
        for (int k = 0; k < Np; ++k) {
            lbx_full(nX + nu*k + 0) = v_min;
            ubx_full(nX + nu*k + 0) = v_max;
            lbx_full(nX + nu*k + 1) = w_min;
            ubx_full(nX + nu*k + 1) = w_max;
        }

        // aggiungo slack variabili 
        lbx_full = cs::DM::vertcat({lbx_full, cs::DM::zeros(ns)});        // s >= 0
        ubx_full = cs::DM::vertcat({ubx_full, cs::DM::ones(ns)*1e20});    // s non limitato superiormente

        lbx_full = cs::DM::vertcat({lbx_full, cs::DM::zeros(ns_obs)});    // s_obs >= 0
        ubx_full = cs::DM::vertcat({ubx_full, cs::DM::zeros(ns_obs)*1e20});

        // ===== Delta input constraints =====
        for (int k = 0; k < Np; ++k) {
            cs::MX vk  = U(nu*k + 0);
            cs::MX wk  = U(nu*k + 1);
            cs::MX v_prev, w_prev;

            if (k == 0) {
                // estrazione input precedente da p
                v_prev = p(weights_start_idx + N_cost_params + 0);
                w_prev = p(weights_start_idx + N_cost_params + 1);
            } else {
                v_prev = U(nu*(k-1) + 0);
                w_prev = U(nu*(k-1) + 1);
            }

            g.push_back(vk - v_prev);
            lbg = cs::DM::vertcat({lbg, -delta_v_max});
            ubg = cs::DM::vertcat({ubg, delta_v_max});

            g.push_back(wk - w_prev);
            lbg = cs::DM::vertcat({lbg, -delta_w_max});
            ubg = cs::DM::vertcat({ubg, delta_w_max});
        }


        cs::MX opt_vars = cs::MX::vertcat(std::vector<cs::MX>{X, U, s, s_obs});


        // ===== NLP setup =====
        std::map<std::string, cs::MX> nlp = {
            {"x", opt_vars},
            {"f", J},
            {"g", cs::MX::vertcat(g)},
            {"p", p}
        };

        // ===== Solver settings =====
        cs::Dict opts;
        opts["ipopt.print_level"] = 0;
        opts["print_time"] = 0;
        opts["ipopt.tol"] = 1e-3;
        opts["ipopt.max_iter"] = 100;

        solver_ = nlpsol("solver", "ipopt", nlp, opts);

        // ===== Inizializzazione precedente =====
        U_previous      = cs::DM::zeros(nu*Np);
        X_previous      = cs::DM::zeros(nx*(Np+1));
        s_previous      = cs::DM::zeros(ns);
        s_obs_previous  = cs::DM::zeros(ns_obs);



        // ROS_INFO("DEBUG sizes: nX=%d, nU=%d, n_opt=%d, ng=%d", nX, nU, n_opt, ng);

    }


    void MpcPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros) {
        if (!initialized_){
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
            
            if (nh_.getParam("/use_warm_start", use_warm_start)) {
                ROS_INFO("MPC: 'use_warm_start' parameter found: %d", use_warm_start);
            } else {
                ROS_WARN("MPC: parameter 'use_warm_start' NOT found in %s! Using default value: true", nh_.getNamespace().c_str());
                use_warm_start = true;
            }

            sub_odom = nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &MpcPlanner::odomCallback, this);
            sub_obs = nh_.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, &MpcPlanner::obstacleGazeboCallback, this);
            sub_mpc_params = nh_.subscribe<mpcParameters>("/mpc/params",1, &MpcPlanner::paramsCallback, this);
            pub_cmd = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
            pub_optimal_traj = nh_.advertise<nav_msgs::Path>("/move_base/TrajectoryPlannerROS/local_plan", 1);
            pub_ref_posearray = nh_.advertise<geometry_msgs::PoseArray>("/pose_array",1);

            clearCostmap_service_client = nh_.serviceClient<std_srvs::Empty>("/move_base_node/clear_costmaps");

            loadParameters();
            
            buildSolver();
            

            initialized_ = true;
            ROS_INFO("MPC local planner initialized");
        }
        else{
            return;
        }
    }

    void MpcPlanner::loadParameters(){
        N_cost_params = 0;
        XmlRpc::XmlRpcValue q_list, r_list, p_list, alfa_list, beta_list;

        if (nh_.getParam("/mpc_planner/Q_weights", q_list)) {
            int Q_size = q_list.size();

            if (Q_size != 3) {
                ROS_WARN("Expected 3 elements for Q, got %d", Q_size);
                return;
            }

            for (int i = 0; i < Q_size; ++i) {

                double value = 0.0;
                if (q_list[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                    value = static_cast<double>(q_list[i]);
                else if (q_list[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
                    value = static_cast<int>(q_list[i]);
                else
                    ROS_WARN("Unexpected type in Q[%d]", i);

                Q(i) = value;
            }
            N_cost_params = N_cost_params + Q_size;
            ROS_INFO("Q loaded");
        }
        else{
            ROS_ERROR("Error loading Q matrix weights");
        }

        if (nh_.getParam("/mpc_planner/R_weights", r_list)) {

            int R_size = r_list.size();
            if (R_size != 2) {
                ROS_WARN("Expected 2 elements for R, got %d", R_size);
                return;
            }

            for (int i = 0; i < R_size; ++i) {
                double value = 0.0;
                if (r_list[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                    value = static_cast<double>(r_list[i]);
                else if (r_list[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
                    value = static_cast<int>(r_list[i]);
                R(i) = value;
            }
            N_cost_params = N_cost_params + R_size;
            ROS_INFO("R loaded");
        }
        else{
            ROS_ERROR("Error loading R matrix weights");
        }


        if (nh_.getParam("/mpc_planner/P_weights", p_list)) {

            int P_size = p_list.size();
            if (P_size != 3) {
                ROS_WARN("Expected 3 elements for P, got %d", P_size);
                return;
            }
        
            for (int i = 0; i < P_size; ++i) {
                double value = 0.0;
                if (p_list[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                    value = static_cast<double>(p_list[i]);
                else if (p_list[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
                    value = static_cast<int>(p_list[i]);
                P(i) = value;
            }
            N_cost_params = N_cost_params + P_size;
            ROS_INFO("P loaded");
        }
        else{
            ROS_ERROR("Error loading P matrix weights");
        }


        if (nh_.getParam("/mpc_planner/alfa", alfa_list)) {

            int yaml_size = alfa_list.size();
    
            // Decidiamo quanto deve essere grande il vettore alfa
            int dimensione_finale;
            if (N_obs > 0) {
                dimensione_finale = N_obs;
            } else {
                dimensione_finale = yaml_size;
            }

            // Ridimensioniamo il vettore Eigen
            alfa.resize(dimensione_finale);

            for(int i=0; i < dimensione_finale; i++){
                // Se abbiamo più ostacoli dei valori nello YAML, 
                // usiamo l'ultimo valore disponibile nello YAML (broadcast)
                int indice_yaml;
                if (i < yaml_size) {
                    indice_yaml = i;
                } else {
                    indice_yaml = yaml_size - 1;
                }

                double value = 0.0;
                if (alfa_list[indice_yaml].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                    value = static_cast<double>(alfa_list[indice_yaml]);
                else if (alfa_list[indice_yaml].getType() == XmlRpc::XmlRpcValue::TypeInt)
                    value = static_cast<int>(alfa_list[indice_yaml]);
                alfa(i) = value;
            }
            N_cost_params = N_cost_params + alfa.size();
            ROS_INFO("alfa loaded");
            std::cout << alfa.size() << std::endl;
        }
        else{
            ROS_ERROR("Error loading alfa weight");
        }

        


        if (nh_.getParam("/mpc_planner/beta", beta_list)) {
            int yaml_size = beta_list.size();
            
            int dimensione_finale;
            if (N_obs > 0) {
                dimensione_finale = N_obs;
            } else {
                dimensione_finale = yaml_size;
            }

            beta.resize(dimensione_finale);

            for(int i = 0; i < dimensione_finale; i++) {
                int indice_yaml;
                if (i < yaml_size) {
                    indice_yaml = i;
                } else {
                    indice_yaml = yaml_size - 1;
                }

                double value = 0.0;
                if (beta_list[indice_yaml].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                    value = static_cast<double>(beta_list[indice_yaml]);
                else if (beta_list[indice_yaml].getType() == XmlRpc::XmlRpcValue::TypeInt)
                    value = static_cast<int>(beta_list[indice_yaml]);
                beta(i) = value;
            }
            N_cost_params = N_cost_params + beta.size();
            ROS_INFO("beta loaded");
            std::cout << beta.size() << std::endl;
        }
        else{
            ROS_ERROR("Error loading beta weight");
        }


        std::cout << "Number of parameters: " << N_cost_params << std::endl;

    }

    void MpcPlanner::paramsCallback(const mpcParameters::ConstPtr& msg){
        Q[0] = msg->Q[0];
        Q[1] = msg->Q[1];
        Q[2] = msg->Q[2];

        R[0] = msg->R[0];
        R[1] = msg->R[1];
        
        P[0] = msg->P[0];
        P[1] = msg->P[1];
        P[2] = msg->P[2];

        // alfa = msg->alfa;
        // beta = msg->beta;

        // std::cout << "Q: " << Q[0] << " " << Q[1] << " " << Q[2] << std::endl;
        // std::cout << "R: " << R[0] << " " << R[1] << " " << std::endl;
        // std::cout << "P: " << P[0] << " " << P[1] << " " << P[2] << std::endl;
        

        for (int i=0; i < msg->objectsList.size(); i++) {
            // std::string object_name = msg->objectsList[i].objectName;
            alfa[i] = msg->objectsList[i].alfa;
            beta[i] = msg->objectsList[i].beta;
            // std::cout << "Alfa_" << i << ": " << alfa[i] << std::endl;
            // std::cout << "Beta_" << i << ": " << beta[i] << std::endl;
        }

        int Q_size = msg->Q.size();
        int R_size = msg->R.size();
        int P_size = msg->P.size();
        int alfa_size = msg->objectsList.size();
        int beta_size = msg->objectsList.size();

        int N_cost_params = Q_size + R_size + P_size + alfa_size + beta_size;
    }


    void MpcPlanner::odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
        /*
            Funzione di callback che viene eseguita quando il plugin si sottoscrive e riceve il messaggio di odometria.
        */

        current_odom_ = *msg;
    }

    void MpcPlanner::obstacleGazeboCallback(const gazebo_msgs::ModelStates::ConstPtr& msg){
        // std::cout << "Gazebo callback" << std::endl;

        try{
            if (obstacles_list.size() != msg->name.size()-3){

                // necessità di ricreare il solver, qualcosa è cambiato.

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

                        Obstacle new_obstacle(x_map, y_map, radius, i, time);
                        obstacles_list.push_back(new_obstacle);
                    }
                }
                N_obs = obstacles_list.size();
                std::cout << "Number of obstacles: " << N_obs << std::endl;
                // Ricarico i parametri per aggiornarne il numero di default e ricreo il solver con le nuove informazioni sugli ostacoli
                loadParameters(); 
                buildSolver();
                std::cout << "Solver built again" << std::endl;
                


            }
            else{
                // non è necessario ricostruire il solver.

                // se l'obstacle_list è vuoto non ci sono ostacoli, quindi non serve fare nulla

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
                        obstacles_list[i].updateInfo(x_map, y_map, time);

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


    bool MpcPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan) {

        /*
            Funzione già presente di default. Viene eseguita:
                - la prima volta quando viene dato il goal;
                - successivamente con la "planner frequency" specificata nel file move_base_params.yaml (interbotix_xslocobot_nav/config)
        */
        
        if(!initialized_)
        {
            ROS_ERROR("This planner has not been initialized");
            return false;
        }

        // std_srvs::Empty srv;

        // Wait for the service to become available
        // if (clearCostmap_service_client.waitForExistence(ros::Duration(5.0))) {
        //     if (clearCostmap_service_client.call(srv)) {
        //         ROS_INFO("Costmaps cleared successfully.");
        //     } else {
        //         ROS_ERROR("Failed to call clear_costmaps service.");
        //     }
        // } else {
        //     ROS_WARN("Service /move_base_node/clear_costmaps not available.");
        // }

        global_plan_.clear();
        global_plan_ = orig_global_plan;

        int global_plan_size = global_plan_.size();

        std::cout << "Global Plan size: " << global_plan_size << std::endl;
        
        
        //quando settiamo un nuovo goal (planner frequency 0 Hz nel config file .yaml -> global planner chiamato una volta, solo all'inizio), 
        //resettiamo il flag. In questo modo in seguito potremo eseguire delle verifiche per capire se il goal è stato raggiunto
        goal_reached_=false;

        //Salviamo quindi solo l'ultimo punto del vettore che contiene tutto il global path
        int size_global_plan=global_plan_.size();
        goal_pose_=global_plan_[size_global_plan-1];

        // Save final goal position and orientation
        goal_pos << goal_pose_.pose.position.x,  goal_pose_.pose.position.y;
        goal_orient = tf2::getYaw(goal_pose_.pose.orientation);

        
        std::cout << "COORDINATE GOAL RICEVUTE: " << std::endl;
        std::cout << "Pose Frame : " << goal_pose_.header.frame_id << std::endl; //FRAME = "map" (coincide con /odom)
        std::cout << "  Coordinates (meters) : " << goal_pos[0] << " " << goal_pos[1] << std::endl;
        std::cout << "  Orientation z-axis (radians) : " << goal_orient << std::endl;


        U_previous = cs::DM::zeros(nu * Np);
        // for (int k = 0; k < Np; ++k) {
        //     U_previous(nu * k + 0) = 0.5; // Una piccola velocità lineare di spinta
        //     U_previous(nu * k + 1) = 0.0;
        // }
    
        X_previous = cs::DM::zeros(nx * (Np + 1)); 
        geometry_msgs::PoseStamped robot_pose_odom;
        robot_pose_odom.header.frame_id = "odom";
        robot_pose_odom.header.stamp = ros::Time(0); // latest available
        robot_pose_odom.pose = current_odom_.pose.pose;

        geometry_msgs::PoseStamped robot_pose_map;
        try {
            tf_->transform(robot_pose_odom, robot_pose_map, "map"); // trasforma in frame map
        } catch (tf2::TransformException &ex) {
            ROS_WARN("Transform from odom to map failed: %s", ex.what());
            return false;
        }

        // ora robot_pose_map.pose.position contiene x,y in map frame
        double x_map = robot_pose_map.pose.position.x;
        double y_map = robot_pose_map.pose.position.y;
        double theta_map = tf2::getYaw(robot_pose_map.pose.orientation);


        // Extract odometry information from the "current_odom_" message
        double x  = x_map;
        double y = y_map;
        double theta = theta_map;

        double delta_theta = angles::shortest_angular_distance(old_theta, theta);
        theta = old_theta + delta_theta;

        old_theta = theta;


        cs::DM current_state = cs::DM::vertcat({x, y, theta});

        // std::cout << "Current state used as warm start:" << std::endl;

        for (int k = 0; k <= Np; ++k) {
            // Assegna il blocco di nx elementi (3: x,y,theta)
            X_previous(cs::Slice(k * nx, k * nx + nx)) = current_state;
            // std::cout << current_state(0) << " " << current_state(1) << " " <<  current_state(2) << std::endl;
        }
        
        // Reset slack variables
        s_previous = cs::DM::zeros(ns);
        s_obs_previous = cs::DM::zeros(Np * N_obs);

        ROS_INFO("MPC Warm Start reset for new plan.");
        
        // Reset flag
        goal_reached_ = false;


        return true;
    }

    bool MpcPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
        if(!initialized_){
            ROS_ERROR("This planner has not been initialized");
            return false;
        }

        if(global_plan_.empty()){
            ROS_ERROR("Global Plan is empty");
            return false;
        }

        geometry_msgs::PoseStamped robot_pose_odom;
        robot_pose_odom.header.frame_id = "odom";
        robot_pose_odom.header.stamp = ros::Time(0); // latest available
        robot_pose_odom.pose = current_odom_.pose.pose;

        geometry_msgs::PoseStamped robot_pose_map;
        try {
            tf_->transform(robot_pose_odom, robot_pose_map, "map"); // trasforma in frame map
        } catch (tf2::TransformException &ex) {
            ROS_WARN("Transform from odom to map failed: %s", ex.what());
            return false;
        }

        // ora robot_pose_map.pose.position contiene x,y in map frame
        double x_map = robot_pose_map.pose.position.x;
        double y_map = robot_pose_map.pose.position.y;
        double theta_map = tf2::getYaw(robot_pose_map.pose.orientation);


        // Extract odometry information from the "current_odom_" message
        double x  = x_map;
        double y = y_map;
        double theta = theta_map;

        double delta_theta = angles::shortest_angular_distance(old_theta, theta);
        theta = old_theta + delta_theta;

        old_theta = theta;

        // Extract odometry information from the "current_odom_" message
        // double x = current_odom_.pose.pose.position.x;
        // double y = current_odom_.pose.pose.position.y;
        // double theta = tf2::getYaw(current_odom_.pose.pose.orientation);

        std::cout << "Actual position of the robot: " << x << " " << y << std::endl;

        double v = current_odom_.twist.twist.linear.x;
        double w = current_odom_.twist.twist.angular.z;


        // Convert pose to state vector (x, y, theta)
        Eigen::Vector3d x_state(x, y, theta);
        Eigen::Vector2d current_rob_pos(x,y);
        double current_rob_orient = theta;

        Eigen::Vector2d dist_from_goal = current_rob_pos - goal_pos;
        
        if(dist_from_goal.norm() <= distance_tolerance){
            std::cout << "------------- DISTANCE REACHED ----------------" << std::endl;
            // Eigen::Vector2d u_opt(0,0);
            cmd_vel.linear.x = 0.0;

            if (std::atan2(std::sin(goal_orient - current_rob_orient), std::cos(goal_orient - current_rob_orient))<=angle_tolerance){
                std::cout<<"ORIENTATION REACHED"<<std::endl;
                std::cout<<"GOAL REACHED"<<std::endl;
                
                cmd_vel.angular.z = 0.0;
                goal_reached_=true;
            }
            else {
                std::cout << "ORIENTATION NOT REACHED" << std::endl;

                cmd_vel.angular.z = 0.5*std::atan2(std::sin(goal_orient - current_rob_orient), std::cos(goal_orient- current_rob_orient));
            }            

        }
        else{
            std::cout << "------------- DISTANCE NOT REACHED ----------------" << std::endl;
            float angle_ = std::atan2((goal_pos[1] - current_rob_pos[1]), (goal_pos[0] - current_rob_pos[0]));

            // if (std::fabs(std::atan2(std::sin(angle_ - current_rob_orient), std::cos(angle_ - current_rob_orient))) >_PI/2){
            //     cmd_vel.linear.x = 0.0;
            //     cmd_vel.angular.z = 0.5*std::atan2(std::sin(angle_ - current_rob_orient), std::cos(angle_- current_rob_orient));

            //     U_previous = cs::DM::zeros(nu * Np); // Input passati azzerati
            //     s_previous = cs::DM::zeros(ns);
            //     s_obs_previous = cs::DM::zeros(Np * N_obs);
                
            //     cs::DM current_state_dm = cs::DM::vertcat({x, y, theta});
            //     X_previous = cs::DM::zeros(nx * (Np + 1));
            //     for (int k = 0; k <= Np; ++k) {
            //         X_previous(cs::Slice(k * nx, k * nx + nx)) = current_state_dm;
            //     }

            // } 
            // else{

                cs::DM p = cs::DM::zeros(nx + nx*(Np+1) + N_cost_params + nu + N_obs*N_obs_info, 1);
                p(0) = x;
                p(1) = y; 
                p(2) = theta;

                buildReferenceTrajectory(p, Np, x, y, theta);


                int weights_start_idx = nx + nx*(Np+1);
                               
                p(weights_start_idx + 0) = Q(0);
                p(weights_start_idx + 1) = Q(1);
                p(weights_start_idx + 2) = Q(2);
                p(weights_start_idx + 3) = R(0);
                p(weights_start_idx + 4) = R(1);
                p(weights_start_idx + 5) = P(0);
                p(weights_start_idx + 6) = P(1);
                p(weights_start_idx + 7) = P(2);


                // --- AGGIORNAMENTO: Alfa e Beta per ogni ostacolo ---
                for (int i = 0; i < N_obs; ++i) {
                    p(weights_start_idx + 8 + i) = alfa(i);            // Vettore alfa
                    p(weights_start_idx + 8 + N_obs + i) = beta(i);    // Vettore beta
                }

                p(weights_start_idx + N_cost_params + 0) = v;
                p(weights_start_idx + N_cost_params + 1) = w;
                
                try
                {            
                    if (N_obs != 0){
                        // std::cout << "Dimension of obstacle_list: " << obstacles_list.size() << std::endl;
                        for (int i=0; i<N_obs; i++){
                            p(weights_start_idx + N_cost_params + nu + N_obs_info*i +0) = obstacles_list[i].pos(0);
                            p(weights_start_idx + N_cost_params + nu + N_obs_info*i +1) = obstacles_list[i].pos(1);
                            p(weights_start_idx + N_cost_params + nu + N_obs_info*i +2) = obstacles_list[i].vel(0);
                            p(weights_start_idx + N_cost_params + nu + N_obs_info*i +3) = obstacles_list[i].vel(1);
                            p(weights_start_idx + N_cost_params + nu + N_obs_info*i +4) = obstacles_list[i].r;

                            // std::cout << "Position of the obstacle: " << obstacles_list[i].pos(0) << "  " << obstacles_list[i].pos(1)  << std::endl;
                            // std::cout << "Velocity of the obstacle: " << obstacles_list[i].vel(0) << "  " << obstacles_list[i].vel(1)  << std::endl;
                            // std::cout << "Radius of the obstacle: " << obstacles_list[i].r << std::endl;
                        }
                    }
                    // std::cout << "obstacle inserted" << std::endl;
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                

                // prepara arg e chiama solver (warm-start)
                std::map<std::string, cs::DM> arg;
                arg["x0"]   = cs::DM::vertcat(std::vector<cs::DM>{X_previous, U_previous, s_previous, s_obs_previous});
                arg["lbx"]  = lbx_full;
                arg["ubx"]  = ubx_full;
                arg["lbg"]  = lbg;
                arg["ubg"]  = ubg;
                arg["p"]    = p;


                // Solve MPC optimization problem
                std::map<std::string, cs::DM> res = solver_(arg);
                if (res.count("x") == 0) {
                    ROS_ERROR("Solver did not return 'x' in result map");
                    return false; // o gestisci fallback
                }

                cs::DM solution;
                try {
                    solution = res.at("x");
                } catch (const std::exception& e) {
                    ROS_ERROR("Exception getting solution: %s", e.what());
                    return false;
                }
                
                auto stats = solver_.stats();
                std::string status = static_cast<std::string>(stats["return_status"]);

                if (status == "Solve_Succeeded" || status == "Feasible_Point_Found" || status == "Maximum_Iterations_Exceeded") {

                    std::cout << "✅ Feasible solution found. State: " << status << std::endl;

                    if (status == "Maximum_Iterations_Exceeded"){
                        ROS_WARN("Maximum_Iterations_Exceeded");
                    }

                    cs::DM X_opt= solution(cs::Slice(0, nx*(Np+1)));                               // primi nx*(Np+1) valori → stati
                    std::cout << "X: " << X_opt.size1() << std::endl;
                    cs::DM U_opt = solution(cs::Slice(nx*(Np+1), nx*(Np+1)+nu*Np));                 // restanti nu*Np valori → controlli
                    std::cout << "U: " << U_opt.size1() << std::endl;
                    cs::DM s_opt = solution(cs::Slice(nx*(Np+1) + nu*Np, nx*(Np+1) + nu*Np + ns));   
                    std::cout << "s: " << s_opt.size1() << std::endl;
                    cs::DM s_obs_opt = solution(cs::Slice(nx*(Np+1) + nu*Np + ns, nx*(Np+1) + nu*Np + ns + Np*N_obs)) ;
                    std::cout << "s_obs: " << s_obs_opt.size1() << std::endl;

                    std::cout << "Dimension of solution: " << (int)solution.size1() << std::endl;


                    double state_cost = 0.0;
                    double input_cost = 0.0;
                    double obstacle_cost = 0.0;
                    double slack_terminal_cost = 0.0;
                    double slack_obs_cost = 0.0;

                    for (int k = 0; k < Np; ++k) {
                        // --- Stati ---
                        double x_ = X_opt(nx*k + 0).scalar();
                        double y_ = X_opt(nx*k + 1).scalar();
                        double th_ = X_opt(nx*k + 2).scalar();

                        double x_r = p(nx + nx*k + 0).scalar();
                        double y_r = p(nx + nx*k + 1).scalar();
                        double th_r = p(nx + nx*k + 2).scalar();

                        double dth_ = std::atan2(std::sin(th_r - th_), std::cos(th_r - th_));

                        state_cost += Q(0) * (x_r - x_)*(x_r - x_)
                                    + Q(1) * (y_r - y_)*(y_r - y_)
                                    + Q(2) * dth_*dth_;

                        // --- Input ---
                        double v = U_opt(nu*k + 0).scalar();
                        double w = U_opt(nu*k + 1).scalar();
                        input_cost += R(0) * v*v + R(1) * w*w;

                        // --- Obstacle ---
                        for (int j = 0; j < N_obs; ++j) {
                            double alfa_j = p(weights_start_idx + 8 + j).scalar();
                            double beta_j = p(weights_start_idx + 8 + N_obs + j).scalar();

                            int idx_obs = j*Np + k;
                            double obs_x = p(weights_start_idx + N_cost_params + nu + N_obs_info*j + 0).scalar();
                            double obs_y = p(weights_start_idx + N_cost_params + nu + N_obs_info*j + 1).scalar();
                            double obs_vx = p(weights_start_idx + N_cost_params + nu + N_obs_info*j + 2).scalar();
                            double obs_vy = p(weights_start_idx + N_cost_params + nu + N_obs_info*j + 3).scalar();

                            double fut_obs_x = obs_x + k*dt*obs_vx;
                            double fut_obs_y = obs_y + k*dt*obs_vy;

                            double dx = x - fut_obs_x;
                            double dy = y - fut_obs_y;
                            double distance = std::sqrt(dx*dx + dy*dy);

                            obstacle_cost += -alfa_j * std::log(beta_j * distance); //alfa/(0.05*(distance * distance));

                            // std::cout << "Distance from obstacle: " << distance << std::endl;

                            // Slack cost per questo ostacolo
                            double s_jk = s_obs_opt(idx_obs).scalar();
                            slack_obs_cost += 1e8 * s_jk * s_jk;
                        }
                    }


                    // --- Terminal cost ---
                    double xN_ = X_opt(nx*Np + 0).scalar();
                    double yN_ = X_opt(nx*Np + 1).scalar();
                    double thN_ = X_opt(nx*Np + 2).scalar();

                    double x_rN = p(nx + nx*Np + 0).scalar();
                    double y_rN = p(nx + nx*Np + 1).scalar();
                    double th_rN = p(nx + nx*Np + 2).scalar();

                    double dth_N_ = std::atan2(std::sin(th_rN - thN_), std::cos(th_rN - thN_));
                    state_cost += P(0) * (x_rN - xN_)*(x_rN - xN_)
                                + P(1) * (y_rN - yN_)*(y_rN - yN_)
                                + P(2) * dth_N_*dth_N_;

                    // --- Terminal slack ---
                    for (int i = 0; i < ns; ++i) {
                        slack_terminal_cost += 30 * s_opt(i).scalar() * s_opt(i).scalar();
                    }

                    // --- Costo totale ---
                    double J_total = state_cost + input_cost + obstacle_cost + slack_terminal_cost + slack_obs_cost;

                    std::cout << "Costo totale: " << J_total << std::endl;
                    std::cout << "Stato: " << state_cost << ", Input: " << input_cost 
                            << ", Ostacoli: " << obstacle_cost 
                            << ", Slack terminale: " << slack_terminal_cost
                            << ", Slack ostacoli: " << slack_obs_cost << std::endl;

                
                    // Extracion of the first control input
                    cmd_vel.linear.x = double(U_opt(0));
                    cmd_vel.angular.z = double(U_opt(1));




                    // std::cout << "U_previous before shift: " << std::endl;
                    // for (int k = 0; k < Np-1; ++k) {
                    //     std::cout << U_previous(nu*k + 0) << std::endl;
                    //     std::cout << U_previous(nu*k + 1) << std::endl;
                    // }

                    // std::cout << "X_previous before shift: " << std::endl;
                    // for (int k = 0; k < Np; ++k) {
                    //     std::cout << X_previous(nx*k + 0) << std::endl;
                    //     std::cout << X_previous(nx*k + 1) << std::endl;
                    //     std::cout << X_previous(nx*k + 2) << std::endl;
                    // }

                    if (use_warm_start){
                        // Shift inputs
                        for (int k = 0; k < Np-1; ++k) {
                            U_previous(nu*k + 0) = U_opt(nu*(k+1) + 0);
                            U_previous(nu*k + 1) = U_opt(nu*(k+1) + 1);
                        }

                        // Ultimo input: ripeti l’ultimo della precedente previsione
                        U_previous(nu*(Np-1) + 0) = U_opt(nu*(Np-1) + 0);
                        U_previous(nu*(Np-1) + 1) = U_opt(nu*(Np-1) + 1);
                        
                        // Shift stati
                        for (int k = 0; k < Np; ++k) {
                            X_previous(nx*k + 0) = X_opt(nx*(k+1) + 0);
                            X_previous(nx*k + 1) = X_opt(nx*(k+1) + 1);
                            X_previous(nx*k + 2) = X_opt(nx*(k+1) + 2);
                        }

                        // Stato terminale: mantieni ultimo
                        X_previous(nx*Np + 0) = X_opt(nx*Np + 0);
                        X_previous(nx*Np + 1) = X_opt(nx*Np + 1);
                        X_previous(nx*Np + 2) = X_opt(nx*Np + 2);

                        // Primo stato della previsione = stato reale corrente del robot
                        X_previous(0) = x;
                        X_previous(1) = y;
                        X_previous(2) = theta;
                    }
                    else{
                        for (int k = 0; k < Np-1; ++k) {
                            U_previous(nu*k + 0) = 0;
                            U_previous(nu*k + 1) = 0;
                        }

                        for (int k = 0; k < Np+1; ++k) {
                            X_previous(nx*k + 0) = x;
                            X_previous(nx*k + 1) = y;
                            X_previous(nx*k + 2) = theta;
                        }
                    }
                    
                    
                    // std::cout << "U_previous after shift: " << std::endl;
                    // for (int k = 0; k < Np-1; ++k) {
                    //     std::cout << U_previous(nu*k + 0) << std::endl;
                    //     std::cout << U_previous(nu*k + 1) << std::endl;
                    // }

                    // std::cout << "X_previous after shift: " << std::endl;
                    // for (int k = 0; k < Np; ++k) {
                    //     std::cout << X_previous(nx*k + 0) << std::endl;
                    //     std::cout << X_previous(nx*k + 1) << std::endl;
                    //     std::cout << X_previous(nx*k + 2) << std::endl;
                    // }

                    s_previous = s_opt;

                    for (int j = 0; j < N_obs; ++j) {
                        for (int k = 0; k < Np-1; ++k) {
                            int idx = j*Np + k;
                            int idx_next = j*Np + (k+1);
                            s_obs_previous(idx) = s_obs_opt(idx_next);
                        }
                        // ultimo step
                        s_obs_previous(j*Np + (Np-1)) = s_obs_opt(j*Np + (Np-1));
                    }
                    
                    // Show the generated optimized path
                    nav_msgs::Path path_msg;
                    path_msg.header.stamp = ros::Time::now();
                    std::string frame = "odom";
                    
                    if (!goal_pose_.header.frame_id.empty()) frame = goal_pose_.header.frame_id;
                    path_msg.header.frame_id = frame;

                    for (int k = 0; k <= Np; ++k) {
                        // indice base nello slice X_opt (X_opt è vettore [x0,y0,th0, x1,y1,th1, ...])
                        int base = nx * k;
                        double xk = double(X_opt(base + 0));
                        double yk = double(X_opt(base + 1));
                        double thk = double(X_opt(base + 2));

                        geometry_msgs::PoseStamped ps;
                        ps.header = path_msg.header;
                        ps.header.stamp = ros::Time::now(); // o la stessa stamp di path_msg
                        ps.pose.position.x = xk;
                        ps.pose.position.y = yk;
                        ps.pose.position.z = 0.0;

                        tf2::Quaternion q;
                        q.setRPY(0.0, 0.0, thk);
                        ps.pose.orientation = tf2::toMsg(q);

                        path_msg.poses.push_back(ps);
                    }

                    // std::cout << "Dimension of path_msg: " << (int)path_msg.poses.size() << std::endl;

                    pub_optimal_traj.publish(path_msg);




                } else {
                    std::cout << "❌ Feasible solution not found. State: " << status << std::endl;
                    // ROS_ERROR("Feasible solution not found");

                    // Stop robot
                    // cmd_vel.linear.x = 0.0;
                    // cmd_vel.angular.z = 0.0;
                    
                    // Reset warm start per tentare un "Cold Start" al prossimo loop
                    U_previous = cs::DM::zeros(nu * Np);
                    X_previous = cs::DM::zeros(nx * (Np + 1)); 
                    // Opzionale: Riempi X_previous con lo stato corrente ripetuto per aiutare il solver
                    for(int k=0; k<=Np; k++){
                        X_previous(cs::Slice(k*nx, k*nx+nx)) = cs::DM::vertcat({x, y, theta});
                    }

                }

            // }
            
        }

        std::cout << "\nVelocity message published\n" << std::endl;
        std::cout << "Linear velocity: " << cmd_vel.linear.x  << std::endl;
        std::cout << "Angular velocity: " << cmd_vel.angular.z  << std::endl;
        std::cout << "\n-------------------------------------------------------------------\n\n\n" << std::endl;

        pub_cmd.publish(cmd_vel);

        return true;
    }

    bool MpcPlanner::isGoalReached() {
        return goal_reached_;
    }

    void MpcPlanner::buildReferenceTrajectory(cs::DM& p_, int Np_, double cur_x, double cur_y, double cur_th) {
        // Trova il punto della traiettoria più vicino al robot
        int closest_idx = 0;
        double min_dist2 = std::numeric_limits<double>::max();

        for (size_t i = 0; i < global_plan_.size(); ++i) {
            double dx = global_plan_[i].pose.position.x - cur_x;
            double dy = global_plan_[i].pose.position.y - cur_y;
            double dist2 = dx*dx + dy*dy;
            if (dist2 < min_dist2) {
                min_dist2 = dist2;
                closest_idx = i;
            }
        }

        double last_theta_ref = cur_th;
        std::cout << "Current orientation: " << last_theta_ref << std::endl;

        // Copia Np punti a partire dal closest_idx
        geometry_msgs::PoseArray ref_pose_array;
        std::string reference_type = "nearest_Np_points";

        if (reference_type == "nearest_Np_points"){
            for (int k = 0; k < Np_+1; ++k) {
                int traj_idx = closest_idx + 10*k;
                double x_ref, y_ref, theta_ref;

                if (traj_idx < global_plan_.size() - 1) {
                    // punto normale (non ultimo)
                    x_ref = global_plan_[traj_idx].pose.position.x;
                    y_ref = global_plan_[traj_idx].pose.position.y;

                    double dx = global_plan_[traj_idx+1].pose.position.x - x_ref;
                    double dy = global_plan_[traj_idx+1].pose.position.y - y_ref;
                    theta_ref = std::atan2(dy, dx); // (-pi; pi]
                }
                else {
                    x_ref = goal_pos[0];
                    y_ref = goal_pos[1];
                    theta_ref = goal_orient;
                }
                
                // THETA REF PRIMA UNWRAP:
                // std::cout << "Theta reference before angle unwrap: " << theta_ref << std::endl;
                
                double angle_diff = angles::shortest_angular_distance(last_theta_ref, theta_ref);
                theta_ref = last_theta_ref + angle_diff;

                // THETA REF DOPO UNWRAP:
                // std::cout << "Theta reference after angle unwrap: " << theta_ref << std::endl;

                p_(nx + nx*k + 0) = x_ref;
                p_(nx + nx*k + 1) = y_ref;
                p_(nx + nx*k + 2) = theta_ref;

                last_theta_ref = theta_ref;

                // For visualization:
                geometry_msgs::Pose pose;
                pose.position.x = x_ref;
                pose.position.y = y_ref;
                pose.position.z = 0.0;

                tf2::Quaternion q;
                q.setRPY(0, 0, theta_ref);
                pose.orientation = tf2::toMsg(q);
                ref_pose_array.poses.push_back(pose);

            }
        }
        else if (reference_type == "equidistant_Np_points"){
            int remaining_points = global_plan_.size() - closest_idx - 1;
            int step = std::max(1, remaining_points / std::max(1, Np_));

            std::cout << "Step: " << step << std::endl;


            for (int k = 0; k < Np_; k++) {
                int traj_idx = std::min(closest_idx + k * step, (int)global_plan_.size() - 2);

                double x_ref = global_plan_[traj_idx].pose.position.x;
                double y_ref = global_plan_[traj_idx].pose.position.y;

                double dx = global_plan_[traj_idx+1].pose.position.x - x_ref;
                double dy = global_plan_[traj_idx+1].pose.position.y - y_ref;

                double theta_ref = std::atan2(dy, dx);

                // THETA REF PRIMA UNWRAP:
                std::cout << "Theta reference before angle unwrap: " << theta_ref << std::endl;
                
                // correzione continuità angolare
                while (theta_ref - last_theta_ref > M_PI)  theta_ref -= 2.0 * M_PI;
                while (theta_ref - last_theta_ref < -M_PI) theta_ref += 2.0 * M_PI;

                // THETA REF DOPO UNWRAP:
                std::cout << "Theta reference after angle unwrap: " << theta_ref << std::endl;

                p_(nx + nx*k + 0) = x_ref;
                p_(nx + nx*k + 1) = y_ref;
                p_(nx + nx*k + 2) = theta_ref;

                geometry_msgs::Pose pose;
                pose.position.x = x_ref;
                pose.position.y = y_ref;
                tf2::Quaternion q;
                q.setRPY(0, 0, theta_ref);
                pose.orientation = tf2::toMsg(q);
                ref_pose_array.poses.push_back(pose);
            }
            p_(nx + nx*Np_ + 0) = goal_pos[0];
            p_(nx + nx*Np_ + 1) = goal_pos[1];
            p_(nx + nx*Np_ + 2) = goal_orient;

            geometry_msgs::Pose pose;
            pose.position.x = goal_pos[0];
            pose.position.y = goal_pos[1];
            pose.position.z = 0.0;

            tf2::Quaternion q;
            q.setRPY(0, 0, goal_orient);
            pose.orientation = tf2::toMsg(q);
            ref_pose_array.poses.push_back(pose);
            }
        
        


        
        ref_pose_array.header.stamp = ros::Time::now();
        ref_pose_array.header.frame_id = "map";  // o il frame corretto per il tuo caso
        pub_ref_posearray.publish(ref_pose_array);
        }

    // void MpcPlanner::loadParameters() {
    //     nh_.param("~control_dt", control_dt_, 0.1);
    //     nh_.param("~horizon", horizon_, 10);
    // }
    


} // namespace mpc_planner