import random
import sys
import os

WORLD_PATH = "/home/ros_ws/src/dual_pathway_model/mpc_planner/world/test.world"

# --- DEFINIZIONE AREE (Min, Max) ---
# A: Robot Spawn (origin)
# C: Robot goal
# B, C, D: For operator spawn

ZONE_A = [[-1.0, 1.0], [-1.0, 1.0]]    
ZONE_B = [[-1.0, 1.0], [8.0, 10.0]]    
ZONE_C = [[8.0, 10.0], [8.0, 10.0]]  
ZONE_D = [[8.0, 10.0], [-1.0, 1.0]]   

class WorldGenerator:
    def __init__(self, seed, n_static_obs = 2):
        self.seed = seed
        random.seed(seed)
        
        # Definiamo le zone in una lista per poterle richiamare tramite indice
        zone_list = [ZONE_A, ZONE_B, ZONE_C, ZONE_D]

        # 1. Scegliamo un indice casuale per lo spawn (escludendo la zona A che è l'indice 0)
        # Quindi scegliamo tra 1 (B), 2 (C), 3 (D)
        spawn_idx = random.choice([1, 2, 3])

        # 2. Definiamo le coppie opposte usando gli indici
        # 0 <-> 2 (A-C)
        # 1 <-> 3 (B-D)
        opposite_map = {
            0: 2, 
            1: 3, 
            2: 0, 
            3: 1
            }

        # 3. Troviamo l'indice del goal
        goal_idx = opposite_map[spawn_idx]

        # 4. Estraiamo le coordinate
        self.spawn_x, self.spawn_y = self.get_random_in_zone(zone_list[spawn_idx])
        self.goal_x, self.goal_y = self.get_random_in_zone(zone_list[goal_idx])
        
        # Generiamo 3 ostacoli statici in posizioni centrali (tra 2m e 7m)
        from_index_to_model_mapping = {
            0:"aws_robomaker_warehouse_ClutteringA_01",
            1:"aws_robomaker_warehouse_ClutteringC_01",
            2:"aws_robomaker_warehouse_ClutteringD_01"
        }

        self.obstacles = [
            [random.uniform(2.0, 7.0), random.uniform(2.0, 7.0), random.uniform(0, 3.14), from_index_to_model_mapping[index]]
            for index in range(n_static_obs)
        ]

    def get_random_in_zone(self, zone):
        x = round(random.uniform(zone[0][0], zone[0][1]), 2)
        y = round(random.uniform(zone[1][0], zone[1][1]), 2)
        return x, y

    def write_sdf_world(self):
        # Generazione XML per gli ostacoli
        obs_xml = ""
        for i, (ox, oy, oyaw, model_name) in enumerate(self.obstacles):
            obs_xml += f"""
    <include>
      <uri>model://{model_name}</uri>
      <name>warehouse_Cluttering_{i}</name>
      <pose>{ox} {oy} 0.1 0 0 {oyaw}</pose> 
    </include>"""

        full_xml = f"""<?xml version="1.0" ?>
<sdf version='1.7'>
  <world name='default'>
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose>0 0 10 0 -0 0</pose>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    {obs_xml}

    <include>
      <uri>model://walls</uri>
      <name>walls</name>
      <pose>4.5 4.5 0.1 0 0 0</pose> 
    </include>

    <include>
      <uri>model://person_walking</uri>
      <name>operator</name>
      <pose>{self.spawn_x} {self.spawn_y} 0.1 0 0 0</pose> 
      <plugin name="operator_controller" filename="libgazebo_ros_planar_move.so">
        <robotNamespace>/operator</robotNamespace>
        <commandTopic>cmd_vel</commandTopic>
        <odometryTopic>odom</odometryTopic>
        <odometryFrame>odom</odometryFrame>
        <odometryRate>20.0</odometryRate>
        <robotBaseFrame>link</robotBaseFrame> 
      </plugin>
    </include>

    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
              <torsional>
                <ode/>
              </torsional>
            </friction>
            <contact>
              <ode/>
            </contact>
            <bounce/>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grey</name>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>
    <wind/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>0</latitude_deg>
      <longitude_deg>0</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>


    <state world_name='default'>
      <sim_time>78 156000000</sim_time>
      <real_time>85 504322791</real_time>
      <wall_time>1742055935 855256578</wall_time>
      <iterations>78156</iterations>
      <model name='ground_plane'>
        <pose>0 0 0 0 -0 0</pose>
        <scale>1 1 1</scale>
        <link name='link'>
          <pose>0 0 0 0 -0 0</pose>
          <velocity>0 0 0 0 -0 0</velocity>
          <acceleration>0 0 0 0 -0 0</acceleration>
          <wrench>0 0 0 0 -0 0</wrench>
        </link>
      </model>

      
      <light name='sun'>
        <pose>0 0 10 0 -0 0</pose>
      </light>
    </state>
    <gui fullscreen='0'>
      <camera name='user_camera'>
        <pose>2.8774 -1.42075 30.0412 -0 1.47564 1.5602</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>
  </world>
</sdf>"""
        
        try:
            with open(WORLD_PATH, 'w') as f:
                f.write(full_xml)
        except Exception as e:
            sys.stderr.write(f"Errore nella scrittura del file: {e}\n")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Errore: specifica un seed.\n")
        sys.exit(1)

    seed_arg = int(sys.argv[1])
    gen = WorldGenerator(seed_arg)
    gen.write_sdf_world()
    
    # LOG su stderr (visibile a terminale ma non catturato da variabili Bash)
    sys.stderr.write(f"[Setup] Seed {seed_arg}: Operator Spawn({gen.spawn_x}, {gen.spawn_y})\n")
    
    # UNICO OUTPUT su stdout: le coordinate del goal per il Bash
    print(f"{gen.goal_x} {gen.goal_y}")