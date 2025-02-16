import gymnasium as gym
import numpy as np

env = gym.make('Ant-v5', xml_file='/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml')

data = []           #Lista para almacenar las tuplas
num_episodes = 10   # Numero de episodios para recolectar datos

for episode in range(num_episodes):
    observation, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Selecciona una acción aleatoria
        print(len(action))
        print(action)
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        data.append((observation, action, next_observation))
        
        # Actualizamos la observacion (Pt a Pt+1)
        observation = next_observation

        done = terminated or truncated

print(f"Total de transiciones recolectadas: {len(data)}")

