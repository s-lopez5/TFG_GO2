import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('/home/santilopez/Documentos/TFG_GO2/model_unitree_go2/scene.xml')
d = mujoco.MjData(m)
sim = mujoco.MjSim(m)

#mujoco.viewer.launch(m, d)


viewer = mujoco.viewer.launch_passive(sim)

# 4. Bucle de simulación
while True:
    # (Opcional) Actualizar o definir una acción.
    # En el entorno Ant, las acciones son vectores continuos. Aquí, generamos una acción aleatoria
    # dentro de los límites definidos en el modelo.
    accion = np.random.uniform(low=m.actuator_ctrlrange[:, 0],
                               high=m.actuator_ctrlrange[:, 1])
    sim.data.ctrl[:] = accion  # Asignar la acción a los actuadores

    # Avanzar la simulación un paso.
    sim.step()

    # Renderizar la simulación en la ventana.
    viewer.render()
