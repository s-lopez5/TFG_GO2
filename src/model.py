import gymnasium
import numpy as np
env = gymnasium.make(
    'Ant-v5',
    xml='~/Documentos/AR-GO2/model_unitree_go2/scene.xml',
    forward_reward_weight=1,    #kept the same as the 'Ant' environment
    ctrl_cost_weight=0.05,     #change because of the stronger motors of 'Go2' robot
    contact_cost_weight=5e-4,  #kept the same as the 'Ant' environment
    healthy_reward=1,   #kept the same as the 'Ant' environment
    main_body=1,    #represent the 'Trunk' of he 'Go2' robot
    healthy_z_range=(0.195, 0.75),  #set to avoid sampling steps where the robot has fallen or jumped too high
    include_cfrc_ext_in_observation=True,   #kept the same as the 'Ant' environment
    exclude_current_positions_from_observation=False,   #kept the same as the 'Ant' environment
    reset_noise_scale=0.1,  #set to avoid policy overfitting 
    frame_skip=25,  #set dt=0.05 ---> dt=frame_skip*model.opt.timestep
    max_episode_steps=1000,  
)
