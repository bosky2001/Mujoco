import mujoco as mj
from mujoco import _functions
from mujoco.glfw import glfw
import numpy as np
from numpy.linalg import inv
import os
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from enum import Enum
import time
xml_path = 'leg2.xml'
simend = 6

step_no = 0


modes = []     
z_height = []
x_height = []

class Params(Enum):
    M_hip = 5
    M_l1 = 0.5
    M_l2 = 0.5
    M_toe = 0.1
    M_total = M_hip + M_l1+ M_l2 + M_toe

    l1 = 1
    l2 = 1
    r1 = 0.05
    r2 = 0.05
    r_toe = 0.07

    I1 = (1/12)* M_l1*(l1**2 + 3*r1**2)
    I2 = (1/12)* M_l2*(l2**2 + 3*r2**2)
    I_toe = (2/5)*M_toe*r_toe**2
    
class Index(Enum):

    x = 0
    z = 1
    q1 = 2
    q2 = 3



mode = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
error_i = np.zeros(2)

def forward_kinematics(model,data):
    q1 = data.qpos[Index.q1.value] 
    q2 = data.qpos[Index.q2.value] 
    
    x = -Params.l1.value * np.sin(q1) - Params.l2.value * np.sin(q1+q2)
    y = 0
    z = -Params.l1.value * np.cos(q1) - Params.l2.value * np.cos(q1+q2)
    return np.array([x,y,z])

def foot_jacobian(model, data):
    J = np.zeros((3, 2))
    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]
    J[0,0] =  -Params.l1.value * np.cos(q1) -  Params.l2.value * np.cos(q1+q2)
    J[0,1] =  -Params.l2.value * np.cos(q1+q2)
    
    J[2,0] =  Params.l1.value * np.sin(q1)+ Params.l2.value * np.sin(q1+q2)
    J[2,1]=   Params.l2.value * np.sin(q1+q2)
    return J

def pid_controller(model, data):
    """
    This function implements a controller that
    mimics the forces of a fixed joint before release
    """
    global error_i
    kp = 0.5*np.array([7,14])
 
    kd = np.array([1,2])
    ki = np.array([0.004, 0.01])*0
    q = np.array([data.qpos[Index.q1.value], data.qpos[Index.q2.value]])
    qdes = np.array([-0.4, 0.8])
    vdes = np.zeros(2)
    v = np.array([data.qvel[Index.q1.value], data.qvel[Index.q2.value]])
    error_i += qdes-q
    tau = np.multiply(kp,qdes-q) + np.multiply(kd,vdes-v) + np.multiply(ki,error_i)

    
    J = foot_jacobian(model, data)
    # print(J)
    # print("and")
    # print(jac_foot)
    # print("------------------------")
    ff = -J.T @ np.array([0,0,-6.5* 9.81])
    data.ctrl[0] = tau[0] - ff[1]
    data.ctrl[1] =  tau[1]- ff[2]

def radial(model, data):
    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value] 


   
    z  = forward_kinematics(model, data)[2]
    x = forward_kinematics(model, data)[0]

    l1 = Params.l1.value
    l2 = Params.l2.value

    r = np.sqrt(l1**2 + l2**2  -2*l1*l2*np.cos(np.pi - q2))

    theta = -np.arctan2(x, -z)
    # print("X:",x, "Z:",z)
    return r, theta

def Jq_r(model, data):
    J = np.zeros((2, 2))

    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]

    l1 = Params.l1.value
    l2 = Params.l2.value

    r,_ = radial(model, data)
    J[0,0] = 0
    J[0,1] = -l1*l2*np.sin(np.pi - q2)/ r

    J[1,0] = 1

    J[1,1] = 1/2
    return J

def slip_flight(model, data, rdes = 1.5, theta_des = 0):
    r, theta = radial(model, data)
    J = Jq_r(model, data)
    vel = J@data.qvel[Index.q1.value:]

    
    #theta_des = -0.1
    kp = 50
    kd = 20
    f = np.array([ kp*(rdes - r) + kd*(-vel[0]) ,
                   30*(theta_des - theta) + 15*(-vel[1])]) 

    g =  np.array([0, 0, -9.81 * Params.M_total.value ])


    tau = J.T @ (f )  
   
    data.ctrl[0] = tau[0]
    data.ctrl[1] = tau[1]

def slip_stance(model, data):
    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]
    z = forward_kinematics(model, data)[2]
    x = forward_kinematics(model, data)[0]
    #zdot = data.qvel[Index.z.value]
    J = Jq_r(model, data)
    dR = J @ data.qvel[Index.q1.value:]
    #nominal height
    
    r,_ = radial(model, data)
    p = 1.5
    w = 23
    phi = np.arctan2( w*(p-r),-dR[0])
    
    beta = 1.5
    ka = 40
    kd = 0
    w_i = 0 #kd*zdot*np.sin(phi)
    #fx = 25*(-x)+ 15*(-dX[0])
    
    J = Jq_r(model, data)
    vel = J@data.qvel[Index.q1.value:]

    u = np.array([ w*w*(p-r)- beta*vel[0] -ka*np.cos(phi), 0])  
    g =  np.array([-9.81 * Params.M_total.value, 0])
    tau = J.T@(u +g) 
    
    data.ctrl[0] = tau[0]
    data.ctrl[1] = tau[1]

def slip_control(model, data):

    tol = 8*1e-2
    stance_h = 1.5
    
    
    #v_flight(model, data, stance_h)
    global mode
    r,theta = radial(model, data)
    vodt =  -foot_jacobian(model, data) @ data.qvel[Index.q1.value:]

    vdot = Jq_r(model, data) @  data.qvel[Index.q1.value:]
    xdot = vdot[0]*np.sin(theta)

    print(vodt)
    print("x is ", xdot)
    delta_z = abs(r-stance_h)
    if mode == 0:
 
        slip_flight(model, data, stance_h)

        if delta_z >= tol:
            
            mode = 1
    
    elif mode == 1:
        slip_stance(model, data)
        if delta_z <= tol:
            mode = 0
    modes.append(mode)
    #z_height.append(radial(model, data)[0])

def v_flight(model,data, zdes= -2 ,vdes=0):
    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]
    X = forward_kinematics(model,data) # [x,0,z]^T
    J = foot_jacobian(model, data)
    dX = J @ data.qvel[Index.q1.value:]
    g =  np.array([0, 0, -9.81 * Params.M_total.value ])
    kp = 250
    kd = 25
    
    fz = np.array([10*(-0.1-X[0]) + 15*(- dX[0]), 0, kp*(zdes-X[2]) + kd*(vdes - dX[2])])
    tau = J.T@( fz)
    # _foot = data.xpos[3, 2]
    # print(_foot)
    data.ctrl[0] = tau[0]
    data.ctrl[1] = tau[1]

def vhad_stance(model, data):
    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]
    z = forward_kinematics(model, data)[2]
    x = forward_kinematics(model, data)[0]
    #zdot = data.qvel[Index.z.value]
    dX = (foot_jacobian(model, data)@data.qvel[Index.q1.value:])
    zdot = dX[2]
    #nominal height
    p = -1.5
    w = 23
    phi = np.arctan2( w*(p-z),-zdot)
    
    beta = 1.5
    ka = 48
    kd = 1
    w_i = 0 #kd*zdot*np.sin(phi)
    fx = 25*(-x)+ 15*(-dX[0])
    u = np.array([0, 0, w*w*(p-z)- beta*zdot -ka*np.cos(phi) ])  
    J = foot_jacobian(model, data)

    g =  np.array([0, 0, -9.81 * Params.M_total.value])
    tau = J.T@(u -g )
    
    data.ctrl[0] = tau[0]
    data.ctrl[1] = tau[1]


def vhad_control(model, data):
    
    tol = 8*1e-2
    stance_h = -1.5
    
    
    #v_flight(model, data, stance_h)
    global mode
    z = forward_kinematics(model, data)[2]
    delta_z = abs(z-stance_h)
    if mode == 0:
 
        v_flight(model, data, stance_h)

        if delta_z >= tol:
            
            mode = 1
    
    elif mode == 1:
        vhad_stance(model, data)
        if delta_z <= tol:
            mode = 0

    modes.append(mode)
    z_height.append(forward_kinematics(model, data)[2])

    
    
def init_controller(model,data):
    
    data.qpos[Index.q1.value] = 0.6
    data.qpos[Index.q2.value] = -1.443
    
    


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

#helper to get constraint momentas
def constraint(data):
    if data.nefc:
        J = data.efc_J.reshape((4,4))
        f = data.efc_force
        return J@f
    else: return np.zeros(4)

def g_force(data):

    G = np.zeros(3)

    g = -9.81
    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]

    l1 = Params.l1.value
    l2 = Params.l2.value

    m_l1 = Params.M_l1.value
    m_l2 = Params.M_l2.value
    m_toe = Params.M_toe.value
    G[0] = g*(Params.M_total.value)
    G[1] = 0.5*g*(l1*(m_l1 + 2*(m_l2+m_toe))*np.sin(q1) + l2*(m_l2+2*m_toe)*np.sin(q1+q2))
    G[2] = 0.5*l2*g*(m_l2 + 2*m_toe)*np.sin(q1+q2)

    return G
    
tau_prev = P_prev =  np.zeros(3)

def get_M(model, data):
    M = np.zeros((model.nq, model.nq))

    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]

    l1 = Params.l1.value
    l2 = Params.l2.value

    m_l1 = Params.M_l1.value
    m_l2 = Params.M_l2.value
    m_toe = Params.M_toe.value

    I_l1 = Params.I1.value
    I_l2 = Params.I2.value
    I_t = Params.I_toe.value

    M[0,0] = Params.M_total.value
    M[0,1] = M[1,0] = 0.5*( l1 * (m_l1 + 2 * (m_l2 + m_toe))*np.sin(q1) \
                      + l2 * (m_l2 + 2*m_toe) * np.sin(q1 + q2))
    
    M[0,2] = M[2,0] = 0.5*l2*(m_l2 + 2*m_toe) * np.sin(q1 + q2) #i have autism

    
    M[1,2] = M[2,1] = I_l2 +I_t+ 0.25*(l2**2)*(m_l2 + 4*m_toe) + 0.5*l1*l2*(m_l2 + 2*m_toe)*np.cos(q2)
    
    M[2, 2] = I_l2 + (l2**2)*(0.25*m_l2 + m_toe)

    M[1, 1] =  I_l1 + I_l2 + I_t + (l1**2)*(0.25*m_l1 + m_l2 + m_toe) \
               + 0.25*(l2**2)*( m_l2 + 4*m_toe) \
               + l1*l2*(m_l2 + 2*m_toe)*np.cos(q2)

    return M
def get_C(model, data):
    C = np.zeros((3,3))
    q1 = data.qpos[Index.q1.value]
    q2 = data.qpos[Index.q2.value]

    q1dot = data.qvel[Index.q1.value]
    q2dot = data.qvel[Index.q2.value]

    l1 = Params.l1.value
    l2 = Params.l2.value

    m_l1 = Params.M_l1.value
    m_l2 = Params.M_l2.value
    m_toe = Params.M_toe.value

    I_l1 = Params.I1.value
    I_l2 = Params.I2.value
    I_t = Params.I_toe.value
    C[0,1] = 0.5*(l1*(m_l1 + 2*(m_l2 + m_toe))*np.cos(q1) \
                  + l2*(m_l2 + 2*m_toe)*np.cos(q1+q2)) *q1dot \
                  + 0.5*l2*(m_l2+2*m_toe)*np.cos(q1 +q2)*q2dot


    C[0,2] =  0.5*l2*(m_l2+2*m_toe)*np.cos(q1 + q2)*(q1dot + q2dot)

    C[1,1] = -0.5*l1*l2*(m_l2+2*m_toe)*np.sin(q2)*q2dot

    C[1,2] = -0.5*l1*l2*(m_l2 + 2*m_toe)*np.sin(q2)*(q1dot + q2dot)

    C[2,1] = 0.5*l1*l2*(m_l2 + 2*m_toe)*np.sin(q2)*q1dot

    return C

def momentum_observer(model,data):

    global P_prev, tau_prev

    #Constructing M from the sparse matrix qM
    #M = np.zeros((model.nv, model.nv))
    #_functions.mj_fullM(model, M, data.qM)
    M = get_M(model, data) #3X3

    #GETTING JACOBIAN
    jac_foot = np.zeros((3, model.nv))
    mj.mj_jacSubtreeCom(model, data, jac_foot, model.body('foot').id)

    C = get_C(model, data)  #3X3
    # C = get_C(model, data) # Cq + G term
    v = np.array([0, data.qvel[1], data.qvel[2]])
    
    #Z joint has no force
    tau_ext = np.array([0, data.actuator_force[0], data.actuator_force[1]]) #3
    
    J = foot_jacobian(model, data)
    P = M@v  # 
    # observer 
    t_delta = 1/1000
    freq = 100 # cut-off frequency 
    gamma = np.exp(-freq*t_delta)
    beta = (1-gamma)/(gamma*t_delta)
    alpha_k = beta*P + tau_ext + C.T@v - g_force(data)
    tau_d = beta*(P -gamma*P_prev) + gamma*(tau_prev)+(gamma-1)*alpha_k  

    tau_prev = tau_d
    P_prev = P

    # S = np.array([[0,1,0],[0,0,1]])

    # J = np.zeros((3,3)) 
    # J[:,0] = foot_jacobian(model, data)[:,0]
    # J[:,2] = foot_jacobian(model, data)[:, 1]

    #contact force from joint torque
    hip_rot = np.array([[np.cos(q),0,np.sin(q)],[0,1,0],[-np.sin(q),0,np.cos(q)]])
    
    contact = np.linalg.pinv ( J.T) @ tau_d[1:]
    return contact


#get the full path

dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path('leg2.xml')  # MuJoCo model
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
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True
# # tweak scales of contact visualization elements
# model.vis.scale.contactwidth = 0.1
# model.vis.scale.contactheight = 0.03
# model.vis.scale.forcewidth = 0.05
# model.vis.map.force = 0.3

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.5])

init_controller(model,data)
#v_flight(model, data, -1.5)
#set the controller

forcetorque = np.zeros(6)
contact_x = []
contact_y = []
contact_z = []
M = np.zeros((model.nv,model.nv))
jac_com = np.zeros((3, model.nv))


obs_x =[]
obs_y =[]
obs_z =[]
# theta1 = []
# theta2 = []

zdes = -1.5
while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        #simulation step
        mj.mj_step(model, data)
        # Apply control
        
        #slip_flight(model, data)
        #slip_stance(model, data)
        slip_control(model, data)

    for j,c in enumerate(data.contact):
        mj.mj_contactForce(model, data, j, forcetorque)

    # q = data.qpos[Index.q1.value]  + data.qpos[Index.q2.value] 
    # toe_frame = np.array([[np.cos(q),0,-np.sin(q)],[0,1,0],[np.sin(q),0,np.cos(q)]])
    # forcetorque_w= toe_frame@forcetorque[0:3]
    
    # contact_x.append(forcetorque_w[0])
    # contact_y.append(forcetorque_w[1])
    # contact_z.append(forcetorque_w[2])
    # rot = Rotation.from_quat(data.xquat[4])
    
    r, theta = radial(model, data)
    x_height.append(r)
    z_height.append(forward_kinematics(model, data)[2])
    # print(rot)
    if (data.time>=simend):
        break;
    
    
    # mom_obs = momentum_observer(model= model, data= data)
    # obs_x.append(mom_obs[0])
    # obs_y.append(mom_obs[1])
    # obs_z.append(mom_obs[2])
    
    # print(mom_obs)
    #print(mode)
    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # print("Mass is {}".format(model.body_mass))
    # M = np.zeros((model.nv, model.nv))
    # _functions.mj_fullM(model, M, data.qM)
    # print(M)
    # print(" And ")
    # print(get_M(model, data))
    # print("----------------------")
    # Show joint frames
    opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1

    # Update scene and render
    cam.lookat[0] = data.qpos[Index.x.value] #camera follows the robot
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

        

glfw.terminate()

# fig,axs = plt.subplots(2)

# axs[0].plot(theta1)
# axs[0].axhline(-0.4,color = 'y', linestyle = '--')
# axs[0].set_title("Theta1")

# axs[1].plot(theta2)
# axs[1].axhline(0.8,color = 'y', linestyle = '--')
# axs[1].set_title("Theta2")
# plt.subplots_adjust(hspace=0.5)
plt.plot(modes, color = 'r', linestyle = '--')
plt.plot(z_height)
#plt.axhline(zdes,color = 'y', linestyle = '--')

# fig, axs = plt.subplots(3)
# axs[0].plot(contact_x)
# axs[0].plot(obs_x)
# axs[0].set_title("X-Contact")

# axs[1].plot(contact_y)
# axs[1].plot(obs_y)
# axs[1].set_title("Y-Contact")

# axs[2].plot(contact_z)
# axs[2].plot(obs_z)
# axs[2].set_title("Z-Contact")
# plt.subplots_adjust(hspace=0.5)
plt.show()
