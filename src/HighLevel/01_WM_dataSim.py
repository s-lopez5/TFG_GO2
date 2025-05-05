import time
import numpy as np
from sdkpy.core.channel import ChannelFactoryInitialize
from sdkpy.core.channel import ChannelPublisher, ChannelSubscriber
from sdkpy.idl.default import unitree_go_msg_dds__SportModeState_
from sdkpy.idl.default import unitree_go_msg_dds__LowState_
from sdkpy.idl.default import unitree_go_msg_dds__LowCmd_
from sdkpy.idl.unitree_go.msg.dds_ import SportModeState_
from sdkpy.idl.unitree_go.msg.dds_ import LowState_
from sdkpy.idl.unitree_go.msg.dds_ import LowCmd_
from sdkpy.utils.crc import CRC

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

def stand_up(cmd, running_time):
    
    phase = np.tanh(runing_time / 1.2)
    for i in range(12):
        cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (
            1 - phase) * stand_down_joint_pos[i]
        cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 3.5
        cmd.motor_cmd[i].tau = 0.0

def walk_forward(cmd, t):
    # Movimiento muy básico para simular avance
    freq = 0.1  # frecuencia de paso (Hz)
    step_amp = 0.2  # amplitud del movimiento

    for i in range(4):  # 4 patas
        hip = i * 3
        knee = hip + 1
        ankle = hip + 2
        #phase_shift = (i % 2) * np.pi  # alternar fase entre patas diagonales
        phase_shift = np.pi/2.0

        cmd.motor_cmd[hip].q = stand_up_joint_pos[hip]
        cmd.motor_cmd[knee].q = stand_up_joint_pos[knee] + step_amp * np.sin(2.0 * np.pi * freq * t)
        cmd.motor_cmd[ankle].q = stand_up_joint_pos[ankle] + step_amp * np.sin(2.0 * np.pi * freq * t + phase_shift)

        cmd.motor_cmd[hip].kp = 50.0
        cmd.motor_cmd[knee].kp = 50.0
        cmd.motor_cmd[ankle].kp = 50.0
        cmd.motor_cmd[hip].kd = 3.5
        cmd.motor_cmd[knee].kd = 3.5
        cmd.motor_cmd[ankle].kd = 3.5
        cmd.motor_cmd[hip].dq = 0.0
        cmd.motor_cmd[knee].dq = 0.0
        cmd.motor_cmd[ankle].dq = 0.0
        cmd.motor_cmd[hip].tau = 0.0
        cmd.motor_cmd[knee].tau = 0.0
        cmd.motor_cmd[ankle].tau = 0.0


if __name__ == "__main__":
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

    # Inicializa todos los motores con 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    # Se levanta el robot
    while runing_time < 2.0:
        step_start = time.perf_counter()
        runing_time += dt

        stand_up(cmd, runing_time)

        cmd.crc = crc.Crc(cmd)
        low_cmd_puber.Write(cmd)

        time.sleep(max(0.0, dt - (time.perf_counter() - step_start)))

    
    # Avanza el robot
    walk_time = 0.5  # segundos para avanzar aprox. 1 metro
    walk_start_time = time.time()

    while time.time() - walk_start_time < walk_time:
        step_start = time.perf_counter()

        walk_forward(cmd, runing_time)
        runing_time += dt

        cmd.crc = crc.Crc(cmd)
        low_cmd_puber.Write(cmd)

        #print(time.time() - walk_start_time)
        # print(cmd.motor_cmd[9].q)
        time.sleep(0.001)
        #time.sleep(max(0.0, dt - (time.perf_counter() - step_start)))

    print("A")
    walk_start_time = time.time()
    while time.time() - walk_start_time < walk_time:
        step_start = time.perf_counter()

        cmd.crc = crc.Crc(cmd)

        #print(time.time() - walk_start_time)
        print(cmd.motor_cmd[9].q)

        time.sleep(0.1)

    print("Movimiento completo.")
    time.sleep(1)
    print('fuera')