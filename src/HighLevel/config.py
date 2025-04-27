ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
# Robot simulation scene file
ROBOT_SCENE = "/home/santilopez/Documentos/unitree_mujoco/unitree_robots/go2/scene.xml"#"../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
# DDS domain id, it is recommended to distinguish from the real  (default is 0 on the real robot)
DOMAIN_ID = 1 # Domain id
# Network interface name, for simulation, it is recommended to use the local loopback "lo"
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 1 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

# Whether to output robot link, joint, sensor information, True for output
PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
# Whether to use virtual tape, 1 to enable
# Mainly used to simulate the hanging process of H1 robot initialization
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

# Simulation time step (unit: s)
# To ensure the reliability of the simulation, it needs to be greater than the time required for viewer.sync() to render once
SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
# Visualization interface runtime step, 0.02 corresponds to 50fps/s
VIEWER_DT = 0.02  # 50 fps for viewer
