#!/usr/bin/env python3

from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion
from numpy import rad2deg

if __name__ == '__main__':
    orientation_q = Quaternion()

    orientation_q.x = -0.0002623399215263191
    orientation_q.y = -0.012660227920718643
    orientation_q.z = 0.7202591230537932
    orientation_q.w = 0.6935895367323109

    print(f"Euler angles: {rad2deg(euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]))}")