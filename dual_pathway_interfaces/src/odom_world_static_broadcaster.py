#!/usr/bin/env python3

import rospy
import tf
import tf.transformations as tr
from geometry_msgs.msg import PoseStamped

class StaticOdomWorldBroadcaster:
    def __init__(self):
        rospy.init_node('static_odom_world_broadcaster')

        self.br = tf.TransformBroadcaster()
        self.static_transform = None
        
        # Sottoscriviamo al topic della posa (es. da un sistema GPS o MoCap)
        self.sub = rospy.Subscriber('/vrpn_client_node/mir/pose', PoseStamped, self.handle_pose)
        
        rospy.loginfo("In attesa della prima posa per bloccare il frame world...")
        
        # Timer per pubblicare la trasformata a 10Hz una volta calcolata
        rospy.Timer(rospy.Duration(0.1), self.publish_fixed_tf)
        rospy.spin()

    def handle_pose(self, msg):
        # Se abbiamo già calcolato la trasformata, ignoriamo i nuovi messaggi
        if self.static_transform is not None:
            return

        rospy.loginfo("Pose received! Publishing transform odom -> world.")

        # 1. Posa del robot nel world
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        ori = [msg.pose.orientation.x, msg.pose.orientation.y, 
               msg.pose.orientation.z, msg.pose.orientation.w]

        # 2. Matrice World -> Robot
        mat_world_robot = tr.concatenate_matrices(tr.translation_matrix(pos), tr.quaternion_matrix(ori))

        # 3. Inversa: Robot -> World (che diventa Odom -> World perché odom=robot all'inizio)
        mat_odom_world = tr.inverse_matrix(mat_world_robot)

        # 4. Salviamo i dati statici
        self.static_transform = {
            'trans': tr.translation_from_matrix(mat_odom_world),
            'quat': tr.quaternion_from_matrix(mat_odom_world)
        }
        
        # Possiamo smettere di ascoltare il topic per risparmiare risorse
        self.sub.unregister()
        rospy.loginfo("Trasformata fissata con successo.")

    def publish_fixed_tf(self, event):
        if self.static_transform is None:
            return

        # Pubblichiamo sempre la stessa trasformata salvata
        self.br.sendTransform(
            self.static_transform['trans'],
            self.static_transform['quat'],
            rospy.Time.now(),
            "world",
            "odom"
        )

if __name__ == '__main__':
    try:
        StaticOdomWorldBroadcaster()
    except Exception as e:
        print(e)