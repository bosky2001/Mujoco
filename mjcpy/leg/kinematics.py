
import mujoco as mj
import numpy as np
from numpy.linalg import inv
from enum import Enum

class Params:
    def __init__(self, make_random=False, std_dev = 0.1):

        self.M_hip = 5
        self.M_l1 = 0.5
        self.M_l2 = 0.5
        self.M_toe = 0.1
        self.M_total = self.M_hip + self.M_l1+ self.M_l2 + self.M_toe

        self.l1 = 1
        self.l2 = 1
        self.r1 = 0.05
        self.r2 = 0.05
        self.r_toe = 0.07

        self.I1 = (1/12)* self.M_l1*(self.l1**2 + 3*self.r1**2)
        self.I2 = (1/12)* self.M_l2*(self.l2**2 + 3*self.r2**2)
        self.I_toe = (2/5)*self.M_toe*self.r_toe**2

        if make_random:
            self.M_hip = self.M_hip * np.random.normal(1, std_dev)
            self.M_l1 = self.M_l1 * np.random.normal(1, std_dev)
            self.M_l2 = self.M_l2 * np.random.normal(1, std_dev)
            self.M_toe = self.M_toe * np.random.normal(1, std_dev)
            self.M_total = self.M_hip + self.M_l1+ self.M_l2 + self.M_toe

            self.l1 = 1*np.random.normal(1, std_dev)
            self.l2 = 1*np.random.normal(1, std_dev)
            self.r1 = 0.05
            self.r2 = 0.05
            self.r_toe = 0.07

            self.I1 = (1/12)* self.M_l1*(self.l1**2 + 3*self.r1**2)
            self.I2 = (1/12)* self.M_l2*(self.l2**2 + 3*self.r2**2)
            self.I_toe = (2/5)*self.M_toe*self.r_toe**2
    
class Index(Enum):

    x = 0
    z = 1
    q1 = 2
    q2 = 3

class Sensor(Enum):

    q1pos = 0
    q1vel = 2
    q2pos = 1
    q2vel = 3

#param = Params()
def forward_kinematics(data, param):
    """
    func to get forward kinematics
    """

    q1 = data.sensordata[Sensor.q1pos.value]
    q2 = data.sensordata[Sensor.q2pos.value]
    
    x = - param.l1 * np.sin(q1) - param.l2 * np.sin(q1+q2)
    y = 0
    z = -param.l1 * np.cos(q1) - param.l2 * np.cos(q1+q2)
    return np.array([x,y,z])

def foot_jacobian(model, data, param):
    '''
    jacobian of toe wrt hip/main body
    '''
    J = np.zeros((3,2))
    q1 = data.sensordata[Sensor.q1pos.value]
    q2 = data.sensordata[Sensor.q2pos.value]
    
    J[0,0] = - param.l1 * np.cos(q1) - param.l2 * np.cos(q1+q2)
    J[0,1] = - param.l2 * np.cos(q1+q2)
    
    J[2,0] =  param.l1 * np.sin(q1)+ param.l2 * np.sin(q1+q2)
    J[2,1]=   param.l2 * np.sin(q1+q2)
    return J

def radial(data, param):
    """
    radial co-ordinates
    """
    q1 = data.sensordata[Sensor.q1pos.value]
    q2 = data.sensordata[Sensor.q2pos.value]

    l1 = param.l1
    l2 = param.l2

    r = np.sqrt(l1**2 + l2**2  -2*l1*l2*np.cos(np.pi - q2))

    theta = q1+ q2/2#-np.arctan2(x, -z)
    # print("X:",x, "Z:",z)
    return r, theta

def Jq_r(model, data, param):
    
    """
    jacobian wrt q in radial
    """
    J = np.zeros((2, 2))
    #q1 = data.sensordata[Sensor.q1pos.value]
    q2 = data.sensordata[Sensor.q2pos.value]

    
    l1 = param.l1
    l2 = param.l2

    r,_ = radial( data, param)
    J[0,0] = 0
    J[0,1] = -l1*l2*np.sin(np.pi - q2)/ r

    J[1,0] = 1

    J[1,1] = 1/2
    return J