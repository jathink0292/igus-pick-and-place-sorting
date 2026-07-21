"""
This module provides a Python interface for controlling the IGUS robot.
It includes classes for representing joint positions, generating commands, and managing the connection to the robot.

It allows users to connect to the robot, send commands to move joints, and receive status updates.

It is just a basic implementation and can be extended with more features as needed.

Author: Lukasz Rojek (lukasz.rojek@srh.de)
Modified: Tolerance-based equality for robot.wait to work correctly.
"""
import socket
import threading
import time
import json

class Joint():
    __ATTRIBUTES = ["A1", "A2", "A3", "A4", "A5", "A6", "E1", "E2", "E3"]

    def __init__(self, *args, **kwargs):

        for attr, value in zip(Joint.__ATTRIBUTES, args):
            setattr(self, attr, float(value))

        for attr in Joint.__ATTRIBUTES[len(args):]:
            setattr(self, attr, 0.0)

        for attr in Joint.__ATTRIBUTES:
            if attr in kwargs:
                setattr(self, attr, float(kwargs[attr]))

    def get_dict(self) -> dict:
        return {
            "A1": self.A1,
            "A2": self.A2,
            "A3": self.A3,
            "A4": self.A4,
            "A5": self.A5,
            "A6": self.A6,
            "E1": self.E1,
            "E2": self.E2,
            "E3": self.E3
        }

    def __eq__(self, other):
        if not isinstance(other, Joint):
            return False
        # Use 1 degree tolerance so robot.wait releases when robot is close enough
        TOLERANCE = 1.0
        for attr in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            if abs(getattr(self, attr) - getattr(other, attr)) > TOLERANCE:
                return False
        return True

    def __str__(self):
        return f"Joint(A1={self.A1}, A2={self.A2}, A3={self.A3}, A4={self.A4}, A5={self.A5}, A6={self.A6}, E1={self.E1}, E2={self.E2}, E3={self.E3})"

class Cart():
    __ATTRIBUTES = ["X", "Y", "Z", "A", "B", "C", "E1", "E2", "E3"]

    def __init__(self, *args, **kwargs):

        for attr, value in zip(Cart.__ATTRIBUTES, args):
            setattr(self, attr, float(value))

        for attr in Cart.__ATTRIBUTES[len(args):]:
            setattr(self, attr, 0.0)

        for attr in Cart.__ATTRIBUTES:
            if attr in kwargs:
                setattr(self, attr, float(kwargs[attr]))

    def get_dict(self) -> dict:
        return {
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "E1": self.E1,
            "E2": self.E2,
            "E3": self.E3
        }

    def __eq__(self, other):
        if not isinstance(other, Cart):
            return False
        # Use 2mm tolerance on X,Y,Z only so robot.wait releases when robot is close enough
        TOLERANCE = 2.0
        for attr in ["X", "Y", "Z"]:
            if abs(getattr(self, attr) - getattr(other, attr)) > TOLERANCE:
                return False
        return True

    def __str__(self):
        return f"Cart(X={self.X}, Y={self.Y}, Z={self.Z}, A={self.A}, B={self.B}, C={self.C}, E1={self.E1}, E2={self.E2}, E3={self.E3})"

class CommandID():
    def __init__(self, start_id: int = 1):
        if not isinstance(start_id, int):
            raise TypeError(f"start_id must be an integer, got {type(start_id).__name__} instead.")
        if start_id < 1 or start_id > 9999:
            raise ValueError(f"start_id must be between 1 and 9999, got {start_id} instead.")
        self.__id = start_id

    def get_id(self):
        self.__increase()
        return self.__id

    def __increase(self):
        if self.__id > 9999:
            self.__id = 0
        else:
            self.__id += 1

class Command():
    __command_id = CommandID()

    @staticmethod
    def move_joint(A1: float | int = 0.0, A2:float | int = 0.0, A3: float | int = 0.0, A4: float | int = 0.0, A5: float | int = 0.0, A6: float | int = 0.0, E1: float | int = 0.0, E2: float | int = 0.0, E3: float | int = 0.0, vel : float | int = 50.0, joint: Joint | None = None):
        if joint is not None:
            return Command.move_joint(**joint.get_dict())
        if not all(isinstance(arg, (float, int)) for arg in [A1, A2, A3, A4, A5, A6, E1, E2, E3]):
            raise TypeError("All joint parameters must be of type float or int.")
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        return f"CRISTART {Command.__command_id.get_id()} CMD Move Joint {A1} {A2} {A3} {A4} {A5} {A6} {E1} {E2} {E3} {vel} CRIEND"

    @staticmethod
    def move_zero(vel: float | int = 50):
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        return Command.move_joint(vel=vel)

    @staticmethod
    def move_L(vel: float | int = 50.0):
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        return Command.move_joint(A3=90.0, A5=90.0, vel=vel)

    @staticmethod
    def move_cart(X: float | int = 0.0, Y:float | int = 0.0, Z: float | int = 0.0, A: float | int = 0.0, B: float | int = 0.0, C: float | int = 0.0, E1: float | int = 0.0, E2: float | int = 0.0, E3: float | int = 0.0, vel : float | int = 50.0, cart: Cart | None = None):
        if cart is not None:
            return Command.move_cart(**cart.get_dict())
        if not all(isinstance(arg, (float, int)) for arg in [X, Y, Z, A, B, C, E1, E2, E3]):
            raise TypeError("All joint parameters must be of type float or int.")
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        return f"CRISTART {Command.__command_id.get_id()} CMD Move Cart {X} {Y} {Z} {A} {B} {C} {E1} {E2} {E3} {vel} CRIEND"

    @staticmethod
    def alive_jog():
        return f"CRISTART {Command.__command_id.get_id()} ALIVEJOG 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 CRIEND"

    @staticmethod
    def connect():
        return f"CRISTART {Command.__command_id.get_id()} CMD Connect CRIEND"

    @staticmethod
    def enable():
        return f"CRISTART {Command.__command_id.get_id()} CMD Enable CRIEND"

    @staticmethod
    def disconnect():
        return f"CRISTART {Command.__command_id.get_id()} CMD Disconnect CRIEND"

    @staticmethod
    def active_multiple_clients():
        return f"CRISTART {Command.__command_id.get_id()} CMD SetActive true CRIEND"

class IGUS(object):

    def __init__(self, host: str = "localhost", port: int = 3921, name:str = "IGUS"):
        self.__host = host
        self.__port = port
        self.__sock = None
        self.__current_joint = None
        self.__set_joint = None
        self.__current_cart = None
        self.__set_cart = None
        self.name = name
        self.wait = False
        self.callback_read_msg = None

    def connect(self):
        self.__sock = socket.create_connection((self.__host, self.__port))
        print(f"{self.name}: initializing communication workers")
        self.__keep_alive()
        self.__keep_reading()
        time.sleep(1)
        print(f"{self.name}: initializing connection")
        self.send(Command.active_multiple_clients())
        self.send(Command.connect())
        self.send(Command.enable())
        time.sleep(3)
        print(f"{self.name}: robot is ready")
        while not self.__current_joint and not self.__current_cart:
            time.sleep(0.01)

    def disconnect(self):
        if self.__sock:
            self.send(Command.disconnect())
            self.__sock.close()
            self.__sock = None

    def send(self, command: str):
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")
        self.__sock.sendall(command.encode('utf-8'))
        time.sleep(0.1)

    def go_to(self, pos: Joint | Cart, vel=50.0):
        if not isinstance(pos, (Joint, Cart)):
            raise TypeError(f"joint must be an instance of Joint or Cart class, {type(pos)} given instead!")

        if isinstance(pos, Joint):
            command = Command.move_joint(**pos.get_dict(), vel=vel)
            self.send(command)
            self.__set_joint = pos
            if self.wait:
                while self.__set_joint != self.__current_joint:
                    time.sleep(0.01)

        if isinstance(pos, Cart):
            command = Command.move_cart(**pos.get_dict(), vel=vel)
            self.send(command)
            self.__set_cart = pos
            if self.wait:
                while self.__set_cart != self.__current_cart:
                    time.sleep(0.01)

    def go_to_zero(self, vel=50.0):
        self.go_to(Joint(), vel=vel)

    def go_to_L(self, vel=50.0):
        self.go_to(Joint(A3=90.0, A5=90.0), vel=vel)

    def __update_status(self, status: str):
        data = status.split(" ")

        if "POSJOINTCURRENT" in data:
            idx = data.index("POSJOINTCURRENT") + 1
            self.__current_joint = Joint(*data[idx:idx + 9])
            if not self.callback_read_msg is None:
                self.callback_read_msg(self.__current_joint)

        if "POSCARTROBOT" in data:
            idx = data.index("POSCARTROBOT") + 1
            self.__current_cart = Cart(*data[idx:idx + 6])
            if not self.callback_read_msg is None:
                self.callback_read_msg(self.__current_cart)

    def __keep_reading(self):
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")

        def read_thread():
            buff = ""
            while True:
                buff = self.__sock.recv(1024).decode("utf-8")
                data_block = buff[buff.find("CRISTART") : buff.rfind("CRIEND") + 6]
                self.__update_status(data_block)

        thread = threading.Thread(target=read_thread)
        thread.daemon = True
        thread.start()

    def __keep_alive(self, interval: float = 0.1):
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")

        def keep_alive_thread():
            while True:
                self.send(Command.alive_jog())
                time.sleep(interval)

        thread = threading.Thread(target=keep_alive_thread)
        thread.daemon = True
        thread.start()

    def read_file(self, filename: str):
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")

        with open("instructions.json", "r") as file:
            data = json.load(file)
            instructions = data.get("instructions", [])
            new_joint_pos = Joint()
            for instruction in instructions:
                if not all(key in instruction for key in ["type", "name", "pos", "vel"]):
                    continue
                if instruction['type'] == "joint":
                    for key, value in instruction["pos"].items():
                        setattr(new_joint_pos, key, value)
                    print(f"executing {instruction['name']} ({instruction['type']}):", new_joint_pos)
                    self.go_to(new_joint_pos, vel=instruction["vel"])
                if instruction['type'] == "base":
                    pass
                if instruction['type'] == "tool":
                    pass

if __name__ == "__main__":
    igus = IGUS()
    igus.wait = True
    try:
        igus.connect()
        igus.go_to_L(vel=100.0)
        igus.go_to(Cart(X=300.0, Z=250.0, A=180.0, B=0.0, C=180.0), vel=100.0)
        igus.go_to(Cart(X=350.0, Z=250.0, A=180.0, B=0.0, C=180.0), vel=100.0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        igus.disconnect()
        print("Disconnected from IGUS robot.")