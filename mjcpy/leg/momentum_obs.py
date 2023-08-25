import mujoco as mj
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from kinematics import *

model = mj.MjModel.from_xml_path('leg2.xml') 
data = mj.MjData(model)  

def g_force(data, param):
    """
    get gravity forces
    """
    G = np.zeros(model.nq)

    g = -9.81
    q1 = data.sensordata[Sensor.q1pos.value]
    q2 = data.sensordata[Sensor.q2pos.value]

    #param = Params()
    l1 = param.l1
    l2 = param.l2

    m_l1 = param.M_l1
    m_l2 = param.M_l2
    m_toe = param.M_toe
    m_total = param.M_total

    G[0] = 0
    G[1] = g*(m_total)
    G[2] = 0.5*g*(l1*(m_l1 + 2*(m_l2+m_toe))*np.sin(q1) + l2*(m_l2+2*m_toe)*np.sin(q1+q2))
    G[3] = 0.5*l2*g*(m_l2 + 2*m_toe)*np.sin(q1+q2)

    return G
    
tau_prev = P_prev =  np.zeros(model.nq)

def get_M(model, data, param):
    """
    Mass matrix for this specific leg type
    """
    M = np.zeros((model.nq, model.nq))

    q1 = data.sensordata[Sensor.q1pos.value]
    q2 = data.sensordata[Sensor.q2pos.value]

    #param = Params()
    l1 = param.l1
    l2 = param.l2

    m_l1 = param.M_l1
    m_l2 = param.M_l2
    m_toe = param.M_toe
    m_total = param.M_total

    I_l1 = param.I1
    I_l2 = param.I2
    I_t = param.I_toe

    M[0,0] =M[1,1] =  m_total
    
     
    M[3, 3] = I_l2 + I_t + (l2**2)*(0.25*m_l2 + m_toe)


    M[2, 2] =  I_l1 + I_l2 + I_t + (l1**2)*(0.25*m_l1 + m_l2 + m_toe) \
               + 0.25*(l2**2)*( m_l2 + 4*m_toe) \
               + l1*l2*(m_l2 + 2*m_toe)*np.cos(q2)

    M[0,2] = M[2,0] = 0.5*(-l1*(m_l1 + 2*(m_l2 + m_toe))*np.cos(q1) \
                           -l2*(m_l2 + 2*m_toe)*np.cos(q1+q2))

    M[2,1] = M[1,2] = 0.5*(l1*(m_l1 + 2*(m_l2 + m_toe))*np.sin(q1) \
                           + l2*(m_l2 + 2*m_toe)*np.sin(q1+q2))
    
    M[0,3] = M[3,0] = -0.5*l2*(m_l2+2*m_toe)*np.cos(q1+q2) #i have autism

    M[1,3] = M[3,1] = 0.5*l2*(m_l2 + 2*m_toe)*np.sin(q1+q2)

    M[2,3] = M[3,2] = I_l2 + I_t + 0.25*l2**2*(m_l2 + 4*m_toe) \
                      + 0.5*l1*l2*(m_l2 + 2*m_toe)*np.cos(q2) 
    
    return M

def get_C(model, data, param):
    """
    Coriolis matrix for this leg type

    """
    C = np.zeros((model.nq,model.nq))
    q1 = data.sensordata[Sensor.q1pos.value]
    q2 = data.sensordata[Sensor.q2pos.value]

    q1dot = data.sensordata[Sensor.q1vel.value]
    q2dot = data.sensordata[Sensor.q2vel.value]

    #param = Params()
    l1 = param.l1
    l2 = param.l2

    m_l1 = param.M_l1
    m_l2 = param.M_l2
    m_toe = param.M_toe

    C[0,2] = 0.5*(l1*(m_l1 + 2*(m_l2 + m_toe))*np.sin(q1) \
                  + l2*(m_l2 + 2*m_toe)*np.sin(q1+q2)) *q1dot \
                  + 0.5*l2*(m_l2+2*m_toe)*np.sin(q1 +q2)*q2dot


    C[0,3] =  0.5*l2*(m_l2+2*m_toe)*np.sin(q1 + q2)*(q1dot + q2dot)

    C[1,2] = 0.5*(l1*(m_l1 +2*(m_l2 + m_toe))*np.cos(q2)  \
                  + l2*(m_l2 + 2*m_toe)*np.cos(q1+q2))*q1dot \
                  + 0.5*l2*(m_l2+2*m_toe)*np.cos(q1 + q2)*q2dot

    C[1,3] = 0.5*l2*(m_l2+2*m_toe)*np.cos(q1 + q2)*(q1dot + q2dot)

    C[2,2] = -0.5*l1*l2*(m_l2+2*m_toe)*np.sin(q2)*q2dot
    C[3,2] = 0.5*l1*l2*(m_l2 + 2*m_toe)*np.sin(q2)*q1dot

    C[2,3] = -0.5*l1*l2*(m_l2 + 2* m_toe)*np.sin(q2) *(q1dot + q2dot)
    

    return C

def momentum_observer(model,data, param):
    """
    big boy momentum observer
    """

    global P_prev, tau_prev

    #Constructing M from the sparse matrix qM
    #M = np.zeros((model.nv, model.nv))
    #_functions.mj_fullM(model, M, data.qM)
    M = get_M(model, data, param = param) #3X3

    #GETTING foot JACOBIAN
    J = foot_jacobian(model, data, param)

    C = get_C(model, data, param)  #3X3
    # C = get_C(model, data) # Cq + G term

    q1dot = data.sensordata[Sensor.q1vel.value]
    q2dot = data.sensordata[Sensor.q2vel.value]
    v = np.array([0,0, data.sensordata[Sensor.q1vel.value], 
                  data.sensordata[Sensor.q2vel.value]])
    
    #X and Z joint has no force
    tau_ext = np.array([0,0, data.actuator_force[0], data.actuator_force[1]]) #3
    
    
    P = M@v  
    # observer 
    t_delta = 1/1000
    freq = 100 # cut-off frequency 
    gamma = np.exp(-freq*t_delta)
    beta = (1-gamma)/(gamma*t_delta)
    alpha_k = beta*P + tau_ext + C.T@v - g_force(data, param)
    tau_d = beta*(P -gamma*P_prev) + gamma*(tau_prev)+(gamma-1)*alpha_k  

    tau_prev = tau_d
    P_prev = P

    # S = np.array([[0,1,0],[0,0,1]])

    # J = np.zeros((3,3)) 
    # J[:,0] = foot_jacobian(model, data)[:,0]
    # J[:,2] = foot_jacobian(model, data)[:, 1]

    #contact force from joint torque
    
    contact = np.linalg.pinv ( J.T) @ tau_d[Index.q1.value:]
    return contact