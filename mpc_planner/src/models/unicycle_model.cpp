#include "mpc_planner/unicycle_model.h"
#include <cmath>
#include <Eigen/Dense>


namespace mpc_planner {

Eigen::VectorXd UnicycleModel::dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) const {
    Eigen::VectorXd x_next(3);
    double theta = x(2);
    double v = u(0);
    double omega = u(1);

    x_next(0) = x(0) + v * std::cos(theta) * dt;
    x_next(1) = x(1) + v * std::sin(theta) * dt;
    x_next(2) = x(2) + omega * dt;

    return x_next;
}

Eigen::MatrixXd UnicycleModel::linearizeA(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) const {
    double theta = x(2);
    double v = u(0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3);

    A(0, 2) = -v * std::sin(theta) * dt;
    A(1, 2) =  v * std::cos(theta) * dt;
    return A;
}

Eigen::MatrixXd UnicycleModel::linearizeB(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double dt) const {
    double theta = x(2);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 2);

    B(0, 0) = std::cos(theta) * dt;
    B(1, 0) = std::sin(theta) * dt;
    B(2, 1) = dt;
    return B;
}

}  // namespace mpc_local_planner
