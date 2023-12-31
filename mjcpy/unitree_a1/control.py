import mujoco as mj
from mujoco import _functions
from mujoco.glfw import glfw

from mujoco import viewer
import numpy as np
from numpy.linalg import inv
import os
import matplotlib.pyplot as plt
from enum import Enum
import copy

import time
xml_path = 'scene.xml'
simend = 30

step_no = 0

class Leg_id:
    def __init__(self, model, leg, Fixed = True):
        if( leg == "FL"):
            self.abd_id = model.joint('FL_hip_joint').id
            self.Upper_id = model.joint('FL_thigh_joint').id
            self.Lower_id = model.joint('FL_calf_joint').id

            self.ctrl_range = [model.actuator("FL_hip").id , model.actuator("FL_thigh").id, model.actuator("FL_calf").id]
        
        if( leg == "RL"):
            self.abd_id = model.joint('RL_hip_joint').id
            self.Upper_id = model.joint('RL_thigh_joint').id
            self.Lower_id = model.joint('RL_calf_joint').id

            self.ctrl_range = [model.actuator("RL_hip").id , model.actuator("RL_thigh").id, model.actuator("RL_calf").id]
        
        if( leg == "FR"):
            self.abd_id = model.joint('FR_hip_joint').id
            self.Upper_id = model.joint('FR_thigh_joint').id
            self.Lower_id = model.joint('FR_calf_joint').id

            self.ctrl_range = [model.actuator("FR_hip").id , model.actuator("FR_thigh").id, model.actuator("FR_calf").id]
        
        if( leg == "RR"):
            self.abd_id = model.joint('RR_hip_joint').id
            self.Upper_id = model.joint('RR_thigh_joint').id
            self.Lower_id = model.joint('RR_calf_joint').id

            self.ctrl_range = [model.actuator("RR_hip").id , model.actuator("RR_thigh").id, model.actuator("RR_calf").id]
        

def a2h(id):
    if id=="FL" or id == "RL":
        return 0.06
    elif id=="FR" or id == "RR":
        return -0.06

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])



# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

stance = -0.28

class flight_ctrl:
    def __init__(self):
        self.nominal_z = stance

    nominal_z = stance
    def FL_pose(self, model, data, leg_id, pose = np.array([0, -0.8, -1.6]) , posevel=np.array([0, 0, 0])):
        abd_des = pose[0]
        Upper_des = pose[1]
        lower_des = pose[2]

        abdvel_des = posevel[0]
        Uppervel_des = posevel[1]
        lowervel_des = posevel[2]

        leg = Leg_id(model, leg_id, Fixed= True)
        
        Kp = np.array([ 2, 2, 2]) 
        Kd = np.array([0.5, 0.5, 0.5])

        p_error = np.array([abd_des - data.qpos[leg.abd_id], 
                            Upper_des - data.qpos[leg.Upper_id],
                            lower_des - data.qpos[leg.Lower_id]])
        
        d_error = np.array([abdvel_des - data.qvel[leg.abd_id], 
                            Uppervel_des - data.qvel[leg.Upper_id],
                            lowervel_des - data.qvel[leg.Lower_id]])
        
        
        
        u = Kp*p_error + Kd*d_error

        tau = u
        
        data.ctrl[leg.ctrl_range[0]] = tau[0]
        data.ctrl[leg.ctrl_range[1]] = tau[1]
        data.ctrl[leg.ctrl_range[2]] = tau[2]
    
    def xyz_pose(self, model, data, leg_id, pose = np.array([0, 0, nominal_z]) , posevel=np.array([0, 0, 0]), ff = np.array([0, 0, 0])):

        X = forward_kinematics2(model, data, leg_id)
        J = Jq_c(model, data, leg_id)

        leg = Leg_id(model, leg_id, Fixed= True)
        # dX = J@data.qvel[leg.abd_id: leg.Lower_id+1]
        dX = get_vel(model, data, leg_id)


        Kp = np.array([ 1000, 1000, 1500]) 
        Kd = np.array([ 10, 10, 10 ])

        # print(X)
        # print(pose)
        p_error = np.array([pose[0]-X[0, 0], pose[1]-X[1, 0], pose[2]-X[2, 0]])

        # print(p_error)
        d_error =  np.array([posevel[0]-dX[0], posevel[1]-dX[1], posevel[2]-dX[2]])
        
        u = Kp*p_error + Kd*d_error

        #np.array([ 0,0,-130/4])
        tau = J.T@ (u + ff)
        # print(tau)
        data.ctrl[leg.ctrl_range[0]] = tau[0]
        data.ctrl[leg.ctrl_range[1]] = tau[1]
        data.ctrl[leg.ctrl_range[2]] = tau[2]



        
spine_phi = []
numerator = []
denom = []
arctan = []

class stance_ctrl:
    def __init__(self):
        self.nominal_z = stance

    nominal_z = stance

    def FL_pose(self, model, data, leg_id, pose = np.array([0, -0.8, -1.6]) , posevel=np.array([0, 0, 0])):
        abd_des = pose[0]
        Upper_des = pose[1]
        lower_des = pose[2]

        abdvel_des = posevel[0]
        Uppervel_des = posevel[1]
        lowervel_des = posevel[2]

        leg = Leg_id(model, leg_id, Fixed= True)
        
        Kp = np.array([ 75, 75, 95]) 
        Kd = np.array([ 1.5, 1.5, 1.5])

        p_error = np.array([abd_des - data.qpos[leg.abd_id], 
                            Upper_des - data.qpos[leg.Upper_id],
                            lower_des - data.qpos[leg.Lower_id]])
        
        d_error = np.array([abdvel_des - data.qvel[leg.abd_id], 
                            Uppervel_des - data.qvel[leg.Upper_id],
                            lowervel_des - data.qvel[leg.Lower_id]])
        
        
        
        u = Kp*p_error + Kd*d_error

        tau = u
        
        data.ctrl[leg.ctrl_range[0]] = tau[0]
        data.ctrl[leg.ctrl_range[1]] = tau[1]
        data.ctrl[leg.ctrl_range[2]] = tau[2]
    
    def xyz_pose(self, model, data, leg_id, pose = np.array([0, 0, nominal_z]) , posevel=np.array([0, 0, 0]), ff = np.array([0, 0, 0])):

        X = forward_kinematics2(model, data, leg_id)
        J = Jq_c(model, data, leg_id)

        leg = Leg_id(model, leg_id, Fixed= True)
        # dX = J@data.qvel[leg.abd_id: leg.Lower_id+1]
        dX = get_vel(model, data, leg_id)


        Kp = np.array([ 100, 300, 1000]) 
        Kd = np.array([ 10, 10, 10 ])

        # print(X)
        # print(pose)
        p_error = np.array([pose[0]-X[0, 0], pose[1]-X[1, 0], pose[2]-X[2, 0]])

        # print(p_error)
        d_error =  np.array([posevel[0]-dX[0], posevel[1]-dX[1], posevel[2]-dX[2]])
        
        u = Kp*p_error + Kd*d_error

        #np.array([ 0,0,-130/4])
        tau = J.T@ (u + ff)
        # print(tau)
        data.ctrl[leg.ctrl_range[0]] = tau[0]
        data.ctrl[leg.ctrl_range[1]] = tau[1]
        data.ctrl[leg.ctrl_range[2]] = tau[2]

        







def get_joint_angles(model, data, id):
    leg = Leg_id(model, id)
    return np.array([data.qpos[leg.abd_id], data.qpos[leg.Upper_id], data.qpos[leg.Lower_id] ])

def get_vel(model, data, id):
    leg = Leg_id(model, id)
    J = Jq_c(model, data, id)

    return J @ data.qvel[leg.abd_id: leg.Lower_id+1]

def theta(model, data, id):

    roll = 0

    alpha = 0

    if id =="FL":
        theta =  alpha + roll

    elif id == "FR":
        theta = alpha + roll
    
    elif id == "RL":
        theta = roll 
    
    elif id == "RR":
        theta = roll 
    
    return theta
    
def forward_kinematics(model, data, id):
    
    [qabd, qhip, qknee] = get_joint_angles(model, data, id)
    l1 = l2 = 0.206
    

    x = -l1*np.sin(qhip) - l2*np.sin(qhip + qknee)
    y = 0
    z = -l1*np.cos(qhip) - l2*np.cos(qhip + qknee)

    Rx = np.matrix([[1, 0, 0], [0, np.cos(qabd), -np.sin(qabd)], [0, np.sin(qabd), np.cos(qabd)]])

    return np.array(Rx@np.array([x,y,z])) 



def forward_kinematics2(model, data, id):
    
    [qabd, qhip, qknee] = get_joint_angles(model, data, id)

    
    l1 = l2 = 0.206
    
    offset_k2t = np.array([0, 0, -l2])
    offset_h2k = np.array([0, 0, -l1])
    offset_a2h = np.array([0, a2h(id), 0])

    offset_b2a = np.array([0.1675, 0.05, 0])
    
    k2t = Ry(qknee) @ offset_k2t
    
    h2t = Ry(qhip) @ (k2t + offset_h2k).T
    
    a2t = Rx(qabd) @ (h2t + np.reshape(offset_a2h,(3,1)))

    qbody = theta(model, data, id)


    return Rx(qbody) @ a2t 

def Jq_c(model, data, id):
    J = np.zeros((3, 3))
    [qabd, qhip, qknee] = get_joint_angles(model, data, id)

    Abd_off = a2h(id)
    # print(f"AbducOffset for {id}: {Abd_off}")
    l1 = l2 = 0.206
    J[0, 0] = 0
    J[0, 1] = -l1*np.cos(qhip) - l2*np.cos(qhip+ qknee)
    J[0, 2] = - l2*np.cos(qhip+ qknee)

    J[1,0] = np.cos(qabd)*(l1*np.cos(qhip) + l2*np.cos(qhip+qknee)) -Abd_off*np.sin(qabd)
    J[1,1] = -np.sin(qabd)*(l1*np.sin(qhip) + l2*np.sin(qhip+qknee))
    J[1,2] = - l2*np.sin(qabd)*np.sin(qhip + qknee)


    J[2, 0] = Abd_off*np.cos(qabd) + (l1*np.cos(qhip) + l2*np.cos(qhip+qknee))*np.sin(qabd)
    J[2, 1] = np.cos(qabd)*(l1*np.sin(qhip) + l2*np.sin(qhip+qknee))
    J[2, 2] = l2*np.cos(qabd) *np.sin(qhip+ qknee)
    return J







delta = []

def clear_ctrl(model, data):
    n = model.nu
    for i in range(n):
        data.ctrl[i] = 0

  
def controller(model, data):
    """
    This function implements a controller that
    mimics the forces of a fixed joint before release
    """

    stance = stance_ctrl()

    stance.FL_pose(model, data, "FL", np.array([0.1, 0.4, -1.]))
    stance.FL_pose(model, data, "RL", np.array([-0.1, 0.8, -1.6]))
    stance.FL_pose(model, data, "FR", np.array([-0.1, 0.8, -1.6]))
    stance.FL_pose(model, data, "RR", np.array([-0.1, 0.8, -1.6]))

    # stance.spine_pd_control(model, data, qdes= 0.2)
    # stance.roll_pd_control(model, data, qdes=-0.2)

    # stance.xyz_pose(model, data, "FL")
    # stance.xyz_pose(model, data, "FR", pose= np.array([ 0, 0, -0.15]))
    # stance.xyz_pose(model, data, "RL", pose= np.array([ 0, 0, -0.15]))
    # stance.xyz_pose(model, data, "RR")

    # # # print(data.ncon)
    # spine_AD(model, data)
    # spine_SLIP(model, data)
    # stance.spine_pd_control(model, data, qdes = 0)
    
    # print(data.ncon)
    
    

   
def leg_init (model, data ,pose, id):
    leg = Leg_id(model, id)
    data.qpos[leg.abd_id] = pose[0]
    data.qpos[leg.Upper_id] = pose[1]
    data.qpos[leg.Lower_id] = pose[2]

def init_controller(model,data):
    # pservo-hip
    leg_init(model, data, np.array([0, 0.8, -1.6]), "FL")
    leg_init(model, data, np.array([-0.1, 0.8, -1.6]), "FR")
    leg_init(model, data, np.array([-0.1, 0.8, -1.6]), "RL")
    leg_init(model, data, np.array([0, 0.8, -1.6]), "RR")
    


def set_position_servo(actuator_no, kp):
    model.actuator_gainprm[actuator_no, 0] = kp
    model.actuator_biasprm[actuator_no, 1] = -kp

def set_velocity_servo(actuator_no, kv):
    model.actuator_gainprm[actuator_no, 0] = kv
    model.actuator_biasprm[actuator_no, 2] = -kv

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#Momentum Observer segment




#get the full path

dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
# xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options



# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)


# initialize visualization data structures
mj.mjv_defaultCamera(cam)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# initialize visualization contact forces
mj.mjv_defaultOption(opt)
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
# opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True
# # tweak scales of contact visualization elements
# model.vis.scale.contactwidth = 0.1
# model.vis.scale.contactheight = 0.03
# model.vis.scale.forcewidth = 0.05
# model.vis.map.force = 0.3

# install GLFW mouse and keyboard callbacks
# glfw.set_key_callback(window, keyboard)
# glfw.set_cursor_pos_callback(window, mouse_move)
# glfw.set_mouse_button_callback(window, mouse_button)
# glfw.set_scroll_callback(window, scroll)

cam.azimuth = 15.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.5])

init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)


with viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() : #and time.time() - start < simend
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mj.mj_step(model, data)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
    #   viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
      viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)