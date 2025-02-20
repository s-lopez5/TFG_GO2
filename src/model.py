import gymnasium as gym
import numpy as np
import pickle
import xml.etree.ElementTree as ET

def modify_box_position(box_path, new_position):
    tree = ET.parse(box_path)
    root = tree.getroot()

    for body in root.findall(".//body[@name='box']"):
        body.set('pos', f"{new_position[0]} {new_position[1]} {new_position[2]}")

    tree.write(box_path)



#Modificar la posicion de la caja Objetivo
box_path = "/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/model_box.xml"
new_position = [0.5, 0, 0.5]
modify_box_position(box_path, new_position)

env = gym.make('Ant-v5', xml_file='/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml', render_mode = "human")

data = []           #Lista para almacenar las tuplas
num_episodes = 10  # Numero de episodios para recolectar datos

for episode in range(num_episodes):
    observation, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Selecciona una acción aleatoria


        next_observation, reward, terminated, truncated, info = env.step(action)
        
        data.append((observation, action, next_observation))
        
        # Actualizamos la observacion (Pt a Pt+1)
        observation = next_observation

        done = terminated or truncated

#Cierra el enviroment
env.close()

print(f"Total de transiciones recolectadas: {len(data)}")
print(f" Tamaño Pt: {len(data[0][0])} \n Tamaño Accion: {len(data[0][1])} \n Tamaño Pt+1: {len(data[0][2])}")

with open("lista_obs.pkl", "wb") as f:
    pickle.dump(data, f)
    