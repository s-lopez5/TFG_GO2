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
    TestOption(name="move forward 1", id=2),   
    TestOption(name="move forward 2", id=3),         
    TestOption(name="rotate (45grad)", id=4),    
    TestOption(name="rotate (60grad)", id=5),
    TestOption(name="rotate (-45grad)", id=6),
    TestOption(name="rotate (-60grad)", id=7),  
    TestOption(name="stop_move", id=8)       
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
    
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    ChannelFactoryInitialize(0, sys.argv[1])

    test_option = TestOption(name=None, id=None) 
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    sport_client = SportClient()  
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    while True:

        user_interface.terminal_handle()

        """
        45 grad = 0.785 rad
        60 grad = 1.047 rad
        90 grad = 1.57 rad
        """

        if test_option.id == 0:
            sport_client.StandUp()
        elif test_option.id == 1:
            sport_client.StandDown()
        elif test_option.id == 2:
            sport_client.Move(1.0,0,0)
        elif test_option.id == 3:
            sport_client.Move(2.0,0,0)
        elif test_option.id == 4:
            sport_client.Move(0,0,0.785)
        elif test_option.id == 5:
            sport_client.Move(0,0,1.047)
        elif test_option.id == 6:
            sport_client.Move(0,0,-0.785)
        elif test_option.id == 7:
            sport_client.Move(0,0,-1.047)
        elif test_option.id == 8:
            sport_client.StopMove()

        time.sleep(1)