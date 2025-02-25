import mujoco as mj
import mujoco.viewer
import numpy as np

model = mj.MjModel.from_xml_path('/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml')
data = mj.MjData(model)

def get_obs(data):
    # Devuelve las observaciones del robot: posiciones y velocidades de las articulaciones.
    qpos = data.qpos.copy()    # Posiciones de las articulaciones
    qvel = data.qvel.copy()    # Velocidades de las articulaciones
    
    return np.concatenate([qpos, qvel]) # Observación completa

# Generar una acción aleatoria dentro de los límites del actuador
def random_action(model):
    # Devuelve una acción aleatoria dentro de los límites de los actuadores.
    act_min = model.actuator_ctrlrange[:, 0]  # Límite inferior
    act_max = model.actuator_ctrlrange[:, 1]  # Límite superior
    
    return np.random.uniform(act_min, act_max)

box_id = model.body("box").id

mj.mj_step(model, data)

pos_cartesiana = data.xpos[box_id]

# Obtener observaciones del estado inicial
observations = get_obs(data)
action = random_action(model)
print(len(pos_cartesiana))
print(pos_cartesiana)
print(len(observations))
print("Observaciones:", observations)
#print(len(action))
#print("Acciones:", action)


