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

        # Parametri dinamica
        self.wn = 10
        self.zeta = 0.9
        self.alpha = 0.5

        # --- NUOVA STRUTTURA DATI ---
        # { 'object_id': {'u_low': 0, 'u_high': 0, 'fear': 0, 'dot_fear': 0, 'last_seen': time} }
        self.tracked_objects = {}
        
        self.previous_time_instant = rospy.get_time()

    def update_low_road_risk(self, msg):
        thalamus_info = json.loads(msg.data)
        rel_thalamus_info = thalamus_info.get("relative_info", {})
        
        # Reset temporaneo degli input low_road per gli oggetti non visti in questo frame
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['u_low'] = 0

        for obj_name, data in rel_thalamus_info.items():
            obs_r = 1 if "rover" in obj_name else 0.3
            rel_dist = data["relative_dist"] - obs_r
            rel_rad_vel = data["radial_vel"]

            # Calcolo Risk (tua logica originale)
            rel_dist_risk = math.exp(-0.5 * ((rel_dist - 0)/2)**2)
            rel_vel_risk = 1 / (1 + math.exp(-0.5 * (-rel_rad_vel - (-1))))
            
            risk_val = round(rel_dist_risk, 3)

            # Inizializza l'oggetto se nuovo, altrimenti aggiorna u_low
            if obj_name not in self.tracked_objects:
                self.tracked_objects[obj_name] = {'u_low': 0, 'u_high': 0, 'fear': 0, 'dot_fear': 0}
            
            self.tracked_objects[obj_name]['u_low'] = risk_val

        self.low_road_risks_pub.publish(json.dumps({k: v['u_low'] for k, v in self.tracked_objects.items()}))

    def update_high_road_risk(self, msg):
        data = json.loads(msg.data)
        objects_list = data.get("objects", [])

        # Reset temporaneo high road
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['u_high'] = 0

        for obj in objects_list:
            name = obj.get("label") # Assicurati che il VLM usi gli stessi nomi del talamo
            if name in self.tracked_objects:
                self.tracked_objects[name]['u_high'] = obj.get("dangerousness", 0)

    def fear_dynamics(self):
        current_time = rospy.get_time()
        dt = current_time - self.previous_time_instant
        self.previous_time_instant = current_time
        
        if dt <= 0: return

        all_fears = {}

        # Calcola la dinamica per OGNI oggetto tracciato
        for obj_id, state in self.tracked_objects.items():
            # Calcolo u_eff per l'oggetto specifico
            u_low = state['u_low']
            u_high = state['u_high']
            u_eff = (u_low + u_high) / 2 if u_high != 0 else u_low

            x1 = state['fear']
            x2 = state['dot_fear']

            if u_eff >= x1: # Fase di crescita (2° ordine)
                dx1 = x2
                dx2 = -2*self.zeta*self.wn*x2 - (self.wn**2)*x1 + (self.wn**2)*u_eff
                x1 += dx1 * dt
                x2 += dx2 * dt
            else: # Fase di rilascio (1° ordine)
                dx1 = -self.alpha * (x1 - u_eff)
                x1 += dx1 * dt
                x2 = dx1

            # Aggiorna lo stato dell'oggetto
            state['fear'] = max(0, min(1.2, round(x1, 3)))
            state['dot_fear'] = round(x2, 3)
            all_fears[obj_id] = state['fear']

        # Pubblica un JSON con i livelli di paura di tutti gli oggetti
        self.fear_level_pub.publish(json.dumps(all_fears))

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.fear_dynamics()
            rate.sleep()

if __name__ == "__main__":
    node = AmygdalaNode()
    node.spin()