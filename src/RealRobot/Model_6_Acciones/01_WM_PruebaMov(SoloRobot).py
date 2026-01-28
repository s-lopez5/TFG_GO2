import time
import sys
from sdkpy.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from sdkpy.idl.default import unitree_go_msg_dds__SportModeState_
from sdkpy.idl.unitree_go.msg.dds_ import SportModeState_
from sdkpy.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import math
from dataclasses import dataclass

@dataclass
class TestOption:
    name: str
    id: int

option_list = [
    TestOption(name="stand_up", id=0),         
    TestOption(name="stand_down", id=1),     
    TestOption(name="move forward", id=2),   
    TestOption(name="Retroceder", id=3),
    TestOption(name="rotete(60grad)", id=4),          
    TestOption(name="rotate (-60grad)", id=5),    
    TestOption(name="StopMove", id=6),
    TestOption(name="FreeWalk", id=7),
    TestOption(name="FreeBound(True)", id=8),  
    TestOption(name="FreeBound(False)", id=9),
    TestOption(name="freeJump(True)", id=10),
    TestOption(name="CrossStep(True)", id=11),
    TestOption(name="ClassikWalk(True)", id=12), 
    TestOption(name="trot run", id=13),
    TestOption(name="StaticWalk", id=14),
    TestOption(name="EconomicGait", id=15),
    TestOption(name="Damp", id=15),
]

class UserInterface:
    def __init__(self):
        self.test_option_ = None

    def convert_to_int(self, input_str):
        try:
            return int(input_str)
        except ValueError:
            return None

    def terminal_handle(self):
        
        input_str = input("\nEnter id (or list): \n")

        if input_str == "list":
            self.test_option_.name = None
            self.test_option_.id = None
            for option in option_list:
                print(f"{option.name}, id: {option.id}")
            return

        for option in option_list:
            if input_str == option.name or self.convert_to_int(input_str) == option.id:
                self.test_option_.name = option.name
                self.test_option_.id = option.id
                print(f"Test: {self.test_option_.name}, test_id: {self.test_option_.id}")
                return

        print("No matching test option found.")

if __name__ == "__main__":

    ChannelFactoryInitialize(0, "eno1")

    test_option = TestOption(name=None, id=None) 
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    sport_client = SportClient()  
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    while True:

        user_interface.terminal_handle()

        print(f"Updated Test Option: Name = {test_option.name}, ID = {test_option.id}\n")

        """
        45 grad = 0.785 rad
        60 grad = 1.047 rad
        90 grad = 1.57 rad
        180 grad = 3.14159 rad
        """

        if test_option.id == 0:
            sport_client.StandUp()
        elif test_option.id == 1:
            sport_client.StandDown()
        elif test_option.id == 2:
            sport_client.Move(0.7,0,0)
        elif test_option.id == 3:
            sport_client.Move(-1.0,0,0)
        elif test_option.id == 4:
            sport_client.Move(0,0,-1.047)
        elif test_option.id == 5:
            sport_client.Move(0,0,0.785)
        elif test_option.id == 6:
            sport_client.StopMove()
        elif test_option.id == 7:
            #No se que hace
            sport_client.Move(0,0,1.57)
        elif test_option.id == 8:
            #Bound run mode(Trotar)
            sport_client.FreeBound(True)
        elif test_option.id == 9:
            #Bound agile mode
            sport_client.FreeBound(False)
        elif test_option.id == 10:
            #Bound jump mode
            sport_client.FreeJump(True)
        elif test_option.id == 11:
            #Modo paso cruzado
            sport_client.CrossStep(True)
        elif test_option.id == 12:
            #Modo marcha clásico        (Este)
            sport_client.ClassicWalk(True)
        elif test_option.id == 13:
            #Entra en el modo de correr normal
            sport_client.TrotRun()
        elif test_option.id == 14:
            #Entra en el modo de caminar normal
            sport_client.StaticWalk()
        elif test_option.id == 15:
            #Entra en el modo resistencia normal
            sport_client.EconomicGait()
        elif test_option.id == 16:
            #Entra en el estado de amortiguación
            sport_client.Damp()
        

        time.sleep(1)



