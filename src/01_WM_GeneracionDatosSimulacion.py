import mujoco as mj
import mujoco.viewer
import time
import numpy as np
import pickle
import xml.etree.ElementTree as ET

def modify_box_position(box_path, new_position):
    tree = ET.parse(box_path)
    root = tree.getroot()

    for body in root.findall(".//body[@name='box']"):
        body.set('pos', f"{new_position[0]} {new_position[1]} {new_position[2]}")

    tree.write(box_path)

def get_box_pos(data, box_id):
    return data.xpos[box_id]

# Generar una acción aleatoria dentro de los límites del actuador
def get_random_action(model):
    # Devuelve una acción aleatoria dentro de los límites de los actuadores.
    act_min = model.actuator_ctrlrange[:, 0]  # Límite inferior
    act_max = model.actuator_ctrlrange[:, 1]  # Límite superior
    
    return np.random.uniform(act_min, act_max)

def get_obs(data):
    # Devuelve las observaciones del robot: posiciones y velocidades de las articulaciones.
    qpos = np.copy(data.qpos)    # Posiciones de las articulaciones
    qvel = np.copy(data.qvel)    # Velocidades de las articulaciones
    
    return np.concatenate([qpos, qvel]) # Observación completa

#Modificar la posicion de la caja Objetivo
"""
box_path = "/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/model_box.xml"
new_position = [0.5, 0, 0.3]
modify_box_position(box_path, new_position)
"""

# Cargamos el xml con mujoco
model = mj.MjModel.from_xml_path('/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml')
data = mj.MjData(model)

trainnig_data = []  #Lista de datos de entrenamiento
num_episodes = 10  # Numero de episodios para recolectar datos

box_id = model.body("box").id   # Cogemos el id correspondiente a la caja objetivo

# Creamos la ventana de visualizacion
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Colocar el robot en la posicion por defect
    data.qpos[:] = model.key_qpos
    mj.mj_step(model, data)
    viewer.sync()   

    for episode in range(num_episodes):
        
        n_iteration = 10000 # Numero maximo de iteraciones realizadas en cada episodio
        for iteration in range(n_iteration):
            obs_t = get_obs(data)  # Obtenemos las observaciones en T
            action = get_random_action(model)   #Obtenemos la accion aleatoria   
            data.ctrl[:] = action   # Aplica la accion
            mj.mj_step(model, data) # Avanza la simulacion
            obs_t1 = get_obs(data) # Obtenemos las observaciones en T+1 

            # Guardar datos
            boxPos = np.copy(get_box_pos(data, box_id))
            datos_entrada = np.concat([np.concat([obs_t, boxPos]), action])
            trainnig_data.append((datos_entrada, np.concat([obs_t1, boxPos]))) 

            viewer.sync()   # Actualiza la visualizacion
            
            # Condicion de parada: si el robot se cae
            robot_height = data.qpos[2]  # Coordenada Z del cuerpo del robot
            if robot_height < 0.1:  # Si la altura es menor a un umbral, el robot esta caido
                print("Simulacion detenida por caida de robot en iteracion", iteration)
                break

        # Reiniciar la simulación al final de cada iteración
        data.qpos[:] = model.key_qpos


print(f"Numero de iteraciones realizadas: {num_episodes}")
print(f"Total de transiciones recolectadas: {len(trainnig_data)}")
print(f" Tamaño Pt+Accion: {len(trainnig_data[0][0])} \n Tamaño Pt+1: {len(trainnig_data[0][1])}")


with open("lista_obs.pkl", "wb") as f:
    pickle.dump(trainnig_data, f)
    