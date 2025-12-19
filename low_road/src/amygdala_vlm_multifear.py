#!/usr/bin/env python3
import rospy
import json
import math
from std_msgs.msg import String

class AmygdalaNode:
    def __init__(self):
        rospy.init_node("amygdala_node", anonymous=True)

        # Subscribers
        self.object_state_sub = rospy.Subscriber("/thalamus/info", String, self.update_low_road_risk)
        self.cortex_input = rospy.Subscriber("/vlm/image/description", String, self.update_high_road_risk)
        
        # Publishers
        self.fear_level_pub = rospy.Publisher("/fearlevel", String, queue_size=1)
        self.low_road_risks_pub = rospy.Publisher("/amygdala/lowroad/risks", String, queue_size=1)
        self.high_road_risks_pub = rospy.Publisher("/amygdala/highroad/risks", String, queue_size=1) # for plotting 

        # Fear dynamics parameters
        self.wn = 10
        self.zeta = 0.9
        self.alpha = 0.5

        # Data structure
        # { 
        #   'object_name': {'u_low': 0, 'u_high': 0, 'fear': 0, 'dot_fear': 0, 'last_seen': time},
        # }
        self.tracked_objects = {}
        
        self.previous_time_instant = rospy.get_time()

    def update_low_road_risk(self, msg):
        """
        Structure of the incoming msg:
        - "object_state": self.object_state,
        - "relative_info": List[Object]

        where Object is:
        "name":{
            relative_dist" 
            "relative_orient" 
            "radial_vel" 
            }
        
        """
        thalamus_info = json.loads(msg.data)
        rel_thalamus_info = thalamus_info["relative_info"]
        
        # Reset of the u_low for each previously detected object
        for object_name in self.tracked_objects:
            self.tracked_objects[object_name]['u_low'] = 0

        # New u_low evaluation for each object
        for object_name, data in rel_thalamus_info.items():
            if "rover" in object_name:
                obs_r = 1 
            else:
                obs_r = 0.3

            rel_dist = data["relative_dist"] - obs_r
            rel_rad_vel = data["radial_vel"]

            # Compute u_low (risk)
            rel_dist_risk = math.exp(-0.5 * ((rel_dist - 0)/2)**2)
            rel_vel_risk = 1 / (1 + math.exp(-0.5 * (-rel_rad_vel - (-1))))
            
            u_low = round(rel_dist_risk*rel_vel_risk, 3) 

            # New object initialization (if not contained in the list)
            if object_name not in self.tracked_objects:
                self.tracked_objects[object_name] = {'u_low': 0, 'u_high': 0, 'fear': 0, 'dot_fear': 0}

            # u_low update
            self.tracked_objects[object_name]['u_low'] = u_low

        self.low_road_risks_pub.publish(json.dumps({k: v['u_low'] for k, v in self.tracked_objects.items()}))

    def update_high_road_risk(self, msg):
        """
        Structure of the incoming msg:
        - summary: str
        - objects: List[Object]

        with Object class:
        - name: str
        - attributes: str
        - relative_distance: float
        - relative_angle: float
        - action: str
        - dangerousness: float
        """

        vlm_info = json.loads(msg.data)
        objects_list = vlm_info["objects"]

        # Reset of the u_high for each previously detected object
        for object_name in self.tracked_objects:
            self.tracked_objects[object_name]['u_high'] = 0

        for object in objects_list:
            name = object["name"]
            if name in self.tracked_objects:
                self.tracked_objects[name]['u_high'] = object["dangerousness"]

        self.high_road_risks_pub.publish(json.dumps({k: v['u_high'] for k, v in self.tracked_objects.items()}))

    def fear_dynamics(self):
        current_time = rospy.get_time()
        dt = current_time - self.previous_time_instant
        self.previous_time_instant = current_time
        
        if dt <= 0: return

        all_fears = {}

        # Fear dyamics for each tracked object
        for obj_id, state in self.tracked_objects.items():
            
            u_low = state['u_low']
            u_high = state['u_high']

            if u_high != 0.0:
                u_eff = (u_low + u_high) / 2
            else:
                u_eff = u_low

            x1 = state['fear']
            x2 = state['dot_fear']

            if u_eff >= x1: 
                dx1 = x2
                dx2 = -2*self.zeta*self.wn*x2 - (self.wn**2)*x1 + (self.wn**2)*u_eff
                x1 += dx1 * dt
                x2 += dx2 * dt
            else: 
                dx1 = -self.alpha * (x1 - u_eff)
                x1 += dx1 * dt
                x2 = dx1

            
            state['fear'] = max(0, min(1.2, round(x1, 3)))
            state['dot_fear'] = round(x2, 3)
            all_fears[obj_id] = state['fear']

        
        self.fear_level_pub.publish(json.dumps(all_fears))

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.fear_dynamics()
            rate.sleep()

if __name__ == "__main__":
    node = AmygdalaNode()
    node.spin()