import pinocchio


import numpy as np
from  
try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    print("This example loads robot descriptions from robot_descriptions.py:")
    print("\n\tpip install robot_descriptions")
 
print("Goal: load a legged robot from its URDF, then modify leg lengths")
 
print("Loading robot description from URDF...")
robot = load_robot_description("upkie_description")
model = robot.model