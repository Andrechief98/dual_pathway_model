#!usr/bin/local/python3
import casadi as ca

def create_model(dt=0.1):
    x = ca.MX.sym('x', 3)    # [x, y, theta]
    u = ca.MX.sym('u', 2)    # [v, omega]

    v, w = u[0], u[1]
    f = ca.vertcat(v * ca.cos(x[2]),
                   v * ca.sin(x[2]),
                   w)

    x_next = x + dt * f
    return ca.Function('f', [x, u], [x_next])