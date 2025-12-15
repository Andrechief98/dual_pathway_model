#!/usr/bin/env python3
import rospy
import math
import tf2_ros
import tf2_geometry_msgs
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point, Quaternion


class ModelMarkerVisualizerTF2:
    def __init__(self):
        rospy.init_node("model_marker_visualizer")

        # Publisher per i marker
        self.marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1)

        # Subscriber a /gazebo/model_states
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_callback)

        # TF2 buffer + listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Parametri di visualizzazione
        self.color_r = 1.0
        self.color_g = 0.0
        self.color_b = 0.0
        self.color_a = 1.0
        self.line_thickness = 0.03
        self.num_points = 60

        rospy.loginfo("Model Marker Visualizer.")
        rospy.spin()

    def model_callback(self, msg):
        marker_array = MarkerArray()

        try:
            # Ottieni la trasformazione map → odom
            transform = self.tf_buffer.lookup_transform(
                "map",  # target frame
                "odom",   # source frame
                rospy.Time(0),
                rospy.Duration(1)
            )
        except Exception as e:
            rospy.logwarn(e)
            return

        for i, name in enumerate(msg.name):
            if name == "ground_plane":
                continue

            pose = msg.pose[i]

            # Costruisci un punto in frame "map"
            pose_map = PoseStamped()
            pose_map.header.frame_id = "map"
            pose_map.header.stamp = rospy.Time.now()
            pose_map.pose = pose


            # print(f"Point in map frame: {point_map.point.x}, {point_map.point.y}, {point_map.point.z}")

            # Trasforma in frame "odom"
            # try:
            #     pose_odom = tf2_geometry_msgs.do_transform_pose(pose_map, transform)
            # except Exception as e:
            #     rospy.logwarn(f"Errore nella trasformazione per {name}: {e}")
            #     continue
            
            # print(f"Point in odom frame: {point_odom.point.x}, {point_odom.point.y}, {point_odom.point.z}")

            # ID stabile
            marker_id = hash(name) % 10000

            if name == "ground_plane" or name == "walls":
                continue
            elif name == "mir":
                marker = self.create_ellipse_marker(pose_map.pose.position, pose_map.pose.orientation, name, marker_id)
            else:
                marker = self.create_circle_marker(pose_map.pose.position, pose_map.pose.orientation, name, marker_id)

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    # --- Marker helper ---
    def create_circle_marker(self, position, orientation, name, marker_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = name
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation = Quaternion(0,0,0,1)
        marker.scale.x = self.line_thickness
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.color_r, self.color_g, self.color_b, self.color_a

        if "rover" in name:
            radius = 1
        else:
            radius = 0.3
            
        for i in range(self.num_points + 1):
            theta = 2 * math.pi * i / self.num_points
            p = Point(x=radius * math.cos(theta), y=radius * math.sin(theta), z=0.05)
            marker.points.append(p)
        marker.lifetime = rospy.Duration(0)
        return marker

    def create_ellipse_marker(self, position, orientation, name, marker_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = name
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation = orientation  # 👉 stessa orientazione del robot
        marker.scale.x = self.line_thickness
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.color_r, self.color_g, self.color_b, self.color_a

        a, b = 0.8, 0.4
        for i in range(self.num_points + 1):
            theta = 2 * math.pi * i / self.num_points
            p = Point(x=a * math.cos(theta), y=b * math.sin(theta), z=0.05)
            marker.points.append(p)
        marker.lifetime = rospy.Duration(0)
        return marker


if __name__ == "__main__":
    try:
        ModelMarkerVisualizerTF2()
    except rospy.ROSInterruptException:
        pass
