import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
import math


def HighStateHandler(msg: SportModeState_):
    #print("Position: ", msg.position)
    #print("Velocity: ", msg.velocity)
    pass


def LowStateHandler(msg: LowState_):
    #print("IMU state: ", msg.imu_state)
    # print("motor[0] state: ", msg.motor_state[0])
    pass


if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")
    hight_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)

    hight_state_suber.Init(HighStateHandler, 10)
    low_state_suber.Init(LowStateHandler, 10)

    low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_puber.Init()
    crc = CRC()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0]=0xFE
    cmd.head[1]=0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0

    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q= 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0
    
    t = 0.0
    freq = 1.0  # Frecuencia de oscilación
    amplitude = 3.0  # Amplitud del torque

    while True:
        # Generamos una onda senoidal para los motores de las patas
        torque = amplitude * math.sin(2 * math.pi * freq * t)

        # Aplicamos el torque a las articulaciones principales (por ejemplo, motores 0, 3, 6, 9: cadera de cada pierna)
        for motor_id in [0, 3, 6, 9]:  # Puedes ajustar según el modelo
            cmd.motor_cmd[motor_id].q = 0.0
            cmd.motor_cmd[motor_id].kp = 0.0
            cmd.motor_cmd[motor_id].dq = 0.0
            cmd.motor_cmd[motor_id].kd = 0.0
            cmd.motor_cmd[motor_id].tau = torque

        cmd.crc = crc.Crc(cmd)
        low_cmd_puber.Write(cmd)

        time.sleep(0.002)
        t += 0.002
    
"""
    while True:
        for i in range(12):
            cmd.motor_cmd[i].q = 0.0 
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0 
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 1.0 
        
        cmd.crc = crc.Crc(cmd)

        #Publish message
        low_cmd_puber.Write(cmd)
        time.sleep(0.002)
        """
