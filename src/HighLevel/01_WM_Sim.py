import mujoco as mj
import mujoco.viewer
import time
import numpy as np

from sdkpy.core.channel import ChannelFactoryInitialize
from sdkpy.core.channel import ChannelPublisher
from sdkpy.idl.unitree_go.msg.dds_ import LowCmd_
from sdkpy.utils.crc import CRC
from sdkpy.idl.default import unitree_go_msg_dds__LowCmd_

import config

stand_up_joint_pos = np.array([
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
],
                              dtype=float)

stand_down_joint_pos = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
],
                                dtype=float)

# Configuración de ganancias del controlador
KP = 50.0   # Ganancia proporcional
KD = 3.5    # Ganancia derivativa

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)

ChannelFactoryInitialize(1, "lo")

low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
low_cmd_puber.Init()

crc = CRC()
dt = 0.002
runing_time = 0.0

cmd = unitree_go_msg_dds__LowCmd_()
cmd.head[0] = 0xFE
cmd.head[1] = 0xEF
cmd.level_flag = 0xFF
cmd.gpio = 0

# Inicializa todos los motores
for i in range(20):
    cmd.motor_cmd[i].mode = 0x01  # Modo servo
    cmd.motor_cmd[i].q = 0.0
    cmd.motor_cmd[i].kp = KP
    cmd.motor_cmd[i].dq = 0.0
    cmd.motor_cmd[i].kd = KD
    cmd.motor_cmd[i].tau = 0.0


# Estados de la máquina de estados
STANDING_UP = 0
STAND_UP = 1
STANDING_DOWN = 2
STAND_DOWN = 3
COMPLETED = 4 

current_state = STAND_DOWN
cycle_completed = False
state_start_time = mj_data.time
transition_duration = 2.0  # segundos para cada transición
standing_duration = 3.0    # tiempo que permanece de pie
lying_duration = 3.0       # tiempo que permanece tumbado

while viewer.is_running() and not cycle_completed:
    current_time = mj_data.time

    # Máquina de estados para controlar las transiciones
    if current_state == STAND_DOWN and current_time - state_start_time > 3.0:
        # Transición de acostado a de pie
        current_state = STANDING_UP
        state_start_time = current_time
        print("Comenzando a levantarse")
    elif current_state == STANDING_UP and current_time - state_start_time > transition_duration:
        # Terminó de levantarse
        current_state = STAND_UP
        state_start_time = current_time
        print("Robot de pie")
    elif current_state == STAND_UP and current_time - state_start_time > 3.0:
        # Transición de pie a acostado
        current_state = STANDING_DOWN
        state_start_time = current_time
        print("Comenzando a acostarse")
    elif current_state == STANDING_DOWN and current_time - state_start_time > transition_duration:
        # Terminó de acostarse
        current_state = COMPLETED  # Cambiamos al estado completado
        cycle_completed = True     # Activamos la condición de parada
        print("Robot acostado")

    # Solo interpolamos si no hemos completado el ciclo
    if current_state != COMPLETED:
        # Interpolación suave entre posiciones
        if current_state == STANDING_UP:
            progress = min(1.0, (current_time - state_start_time) / transition_duration)
            target_positions = [
                stand_down_joint_pos[i] + (stand_up_joint_pos[i] - stand_down_joint_pos[i]) * progress
                for i in range(12)
            ]
        elif current_state == STANDING_DOWN:
            progress = min(1.0, (current_time - state_start_time) / transition_duration)
            target_positions = [
                stand_up_joint_pos[i] + (stand_down_joint_pos[i] - stand_up_joint_pos[i]) * progress
                for i in range(12)
            ]
        elif current_state == STAND_UP:
            target_positions = stand_up_joint_pos
        else:  # STAND_DOWN
            target_positions = stand_down_joint_pos

        # Configurar comandos de motor
        for i in range(12):  # Solo los 12 motores de las piernas
            cmd.motor_cmd[i].q = target_positions[i]
            cmd.motor_cmd[i].kp = KP
            cmd.motor_cmd[i].kd = KD
        
        # Calcular CRC y enviar comando
        cmd.crc = crc.Crc(cmd)
        low_cmd_puber.Write(cmd)
    
    # Simulación de un paso
    mujoco.mj_step(mj_model, mj_data)

    # Imprimir posición del robot cada 500 pasos (~1 segundo si timestep es 0.002)
    if mj_data.time % 0.5 < config.SIMULATE_DT:
        print("qpos:", mj_data.qpos[:10])  # Mostrar primeras 10 por claridad

    # Sincronizar la visualización
    viewer.sync()
    time.sleep(0.001)  # Pequeña pausa para evitar sobrecarga

# Mensaje final
print("Simulación completada")