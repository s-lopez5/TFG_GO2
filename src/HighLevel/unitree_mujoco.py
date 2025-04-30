import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from sdkpy.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)



def SimulationThread():
    try:
        global mj_data, mj_model
        
        ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
        #print("Inicializando UnitreeSdk2Bridge con:")
        #print("  mj_model bodies:", [mj_model.body(i).name for i in range(mj_model.nbody)])
        #print("  mj_model joints:", [mj_model.joint(i).name for i in range(mj_model.njnt)])
        unitree = UnitreeSdk2Bridge(mj_model, mj_data)
  
        step_counter = 0
        
        while viewer.is_running():
            
            step_start = time.perf_counter()
            
            locker.acquire()

            mujoco.mj_step(mj_model, mj_data)

            # Imprimir posición del robot cada 500 pasos (~1 segundo si timestep es 0.002)
            if step_counter % 500 == 0:
                print("qpos:", mj_data.qpos[:10])  # Mostrar primeras 10 por claridad

            step_counter += 1

            locker.release()

            time_until_next_step = mj_model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except Exception as e:
        print("Error en SimulateThread:", e)


def PhysicsViewerThread():

    try:
            while viewer.is_running():
                locker.acquire()
                viewer.sync()
                locker.release()
                time.sleep(config.VIEWER_DT)
    except Exception as e:
        print("Error en PhysicsViewerThread:", e)

if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
