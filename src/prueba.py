import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

"""
env = gym.make('Ant-v5')
#env = gym.make('Ant-v5', xml_file='/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml')


data = []           #Lista para almacenar las tuplas
num_episodes = 10   # Numero de episodios para recolectar datos

for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    #i = 0
    #while i < 10:
    while not done:
        action = env.action_space.sample()  # Selecciona una acción aleatoria
        print("Acción muestreada:", action)
        next_observation, reward, terminated, truncated, info = env.step(action)
        print("obs muestreada:", next_observation)
        print(len(next_observation))
        
        data.append((observation, action, next_observation))
        
        # Actualizamos la observacion (Pt a Pt+1)
        observation = next_observation

        done = terminated or truncated
        #i += 1

print(f"Total de transiciones recolectadas: {len(data)}")
"""

#m = mujoco.MjModel.from_xml_path('/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml')
#d = mujoco.MjData(m)

#mujoco.viewer.launch(m, d)



# 2. Cargar el modelo y crear la simulación
m = mujoco.MjModel.from_xml_path('/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml')
d = mujoco.MjData(m)

accion = np.random.uniform(low=m.actuator_ctrlrange[:, 0],
                               high=m.actuator_ctrlrange[:, 1])

print(len(accion))
print(accion)

