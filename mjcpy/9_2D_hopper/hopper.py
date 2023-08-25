import mujoco as mj
from mujoco import _functions
from mujoco.glfw import glfw
import numpy as np
from numpy.linalg import inv
import os
import matplotlib.pyplot as plt

xml_path = 'hopper.xml'
simend = 20

step_no = 0

FSM_AIR1 = 0
FSM_STANCE1 = 1
FSM_STANCE2 = 2
FSM_AIR2 = 3

fsm = FSM_AIR1

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def controller(model, data):
    """
    This function implements a controller that
    mimics the forces of a fixed joint before release
    """
    global fsm
    global step_no

    body_no = 3
    z_foot = data.xpos[body_no, 2]
    vz_torso = data.qvel[1]

    # Lands on the ground
    if fsm == FSM_AIR1 and z_foot < 0.05:
        fsm = FSM_STANCE1

    # Moving upward
    if fsm == FSM_STANCE1 and vz_torso > 0.0:
        fsm = FSM_STANCE2

    # Take off
    if fsm == FSM_STANCE2 and z_foot > 0.05:
        fsm = FSM_AIR2

    # Moving downward
    if fsm == FSM_AIR2 and vz_torso < 0.0:
        fsm = FSM_AIR1
        step_no += 1

    if fsm == FSM_AIR1:
        set_position_servo(2, 100)
        set_velocity_servo(3, 10)

    if fsm == FSM_STANCE1:
        set_position_servo(2, 1000)
        set_velocity_servo(3, 0)

    if fsm == FSM_STANCE2:
        set_position_servo(2, 1000)
        set_velocity_servo(3, 0)
        data.ctrl[0] = -0.2

    if fsm == FSM_AIR2:
        set_position_servo(2, 100)
        set_velocity_servo(3, 10)
        data.ctrl[0] = 0.0

def init_controller(model,data):
    # pservo-hip
    set_position_servo(0, 100)

    # vservo-hip
    set_velocity_servo(1, 10)

    # pservo-knee
    set_position_servo(2, 1000)

    # vservo-knee
    set_velocity_servo(3, 0)

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

#helper to get constraint momentas
def constraint(data):
    if data.nefc:
        J = data.efc_J.reshape((4,4))
        f = data.efc_force
        return J@f
    else: return np.zeros(4)
def g_force(data):
    mhip = 1
    mleg = 1
    mtoe= 0.1
    g = 9.81
    d1 = 1
    d2 = 0.25
    q = data.qpos[2]
    l = data.qpos[3]

    return np.array([0,g*(mtoe+mhip+ mleg), g*(d2*mtoe +d1*(mleg+mtoe) +mtoe*l)
                     *np.sin(q), -g*mtoe*np.cos(q)])
    
tau_prev = P_prev =  np.zeros(4)

def get_C(model, data):
    mhip = 1
    mleg = 1
    mtoe= 0.1
    g = 9.81
    d1 = 1
    d2 = 0.25
    q = data.qpos[2]
    l = data.qpos[3]
    qdot = data.qvel[2]
    ldot = data.qvel[3]
    C = np.zeros((model.nv, model.nv))

    C[0,2] = mtoe*np.cos(q)*ldot - (d2*mtoe+d1*(mleg+mtoe)+mtoe*l)*np.sin(q)*qdot
    C[0,3] = mtoe*np.cos(q)*qdot
    C[1,3] = mtoe*np.sin(q)*qdot
    C[1,2] = mtoe*np.sin(q)*ldot + np.cos(q)*(d2*mtoe + d1*(mleg + mtoe) + mtoe*l) *qdot
    C[2,2] = 0.5*(mtoe*l + mtoe*(2*(d1+d2)+l))*ldot
    C[2,3] = 0.5*(mtoe*l + mtoe*(2*(d1+d2)+l))*qdot
    C[3,2] = 0.5*(mtoe*l + mtoe*(2*(d1+d2)+l))*qdot
    return C

def jacobian_toe(model,data):
    
    d1 = 1
    d2 = 0.25
    q = data.qpos[2]
    l = data.qpos[3]
    qdot = data.qvel[2]
    ldot = data.qvel[3]

    J = np.zeros((3,model.nv))
    J[0,0] = 1
    J[-1,1]= 1
    J[0,2] = np.cos(q)*(d1+d2 +l)
    J[0,3] = np.sin(q)

    J[-1,2] = np.sin(q)*(d1+d2 +l)
    J[-1,3] = -np.cos(q)

    return J
def momentum_observer(model,data):

    global P_prev, tau_prev

    #Constructing M from the sparse matrix qM
    M = np.zeros((model.nv, model.nv))
    _functions.mj_fullM(model, M, data.qM)

    #GETTING JACOBIAN
    jac_foot = np.zeros((3, model.nv))
    mj.mj_jacSubtreeCom(model, data, jac_foot, model.body('foot').id)

    C = get_C(model, data) 
    # C = get_C(model, data) # Cq + G term
    v = data.qvel
    tau_ext = data.actuator_force
    P = M@v
    
    # observer 
    t_delta = 0.001
    freq = 100 # cut-off frequency 
    gamma = np.exp(-freq*t_delta)
    beta = (1-gamma)/(gamma*t_delta)
    alpha_k = beta*P + tau_ext + C.T@data.qvel - g_force(data)
    tau_d = beta*(P -gamma*P_prev) + gamma*(tau_prev)+(gamma-1)*alpha_k  

    tau_prev = tau_d
    P_prev = P

    

    #contact force from joint torque
    # hip_rot = np.array([[np.cos(q),0,np.sin(q)],[0,1,0],[-np.sin(q),0,np.cos(q)]])
    J = jacobian_toe(model, data)
    contact = np.linalg.pinv(J.T) @ tau_d
    return contact


#get the full path

dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path('hopper.xml')  # MuJoCo model
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

#set the controller
mj.set_mjcb_control(controller)
forcetorque = np.zeros(6)
contact_x = []
contact_y = []
contact_z = []
M = np.zeros((model.nv,model.nv))
jac_com = np.zeros((3, model.nv))

obs_x =[]
obs_y =[]
obs_z =[]
last_time = 0
while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/1000.0):
        mj.mj_step(model, data)
    
     #contact forces, something has to be
    
    #print(np.linalg.pinv(jac_com))
    for j,c in enumerate(data.contact):
        mj.mj_contactForce(model, data, j, forcetorque)
    
    # q = data.qpos[2]
    # hip_rot = np.array([[np.sin(q),0,-np.cos(q)],[0,1,0],[np.cos(q),0,np.sin(q)]])
    q= data.qpos[2]
    hip_rot = np.array([[np.sin(q),0,-np.cos(q)],[0,-1,0],[np.cos(q),0,np.sin(q)]])
    forcetorque_w= hip_rot@forcetorque[0:3]
    
    contact_x.append(forcetorque_w[0])
    contact_y.append(forcetorque_w[1])
    contact_z.append(forcetorque_w[2])
    mom_obs = momentum_observer(model= model, data= data)
    obs_x.append(mom_obs[0])
    obs_y.append(mom_obs[1])
    obs_z.append(mom_obs[2])

    #print(data.actuator_force)
    
    if (data.time>=simend):
        break;
    if (data.time - last_time)>1/60.0:
        last_time = data.time
        M = np.zeros((model.nv, model.nv))
        _functions.mj_fullM(model, M, data.qM)
        print(M@data.qvel)

        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        cam.lookat[0] = data.qpos[0] #camera will follow qpos
        mj.mjv_updateScene(model, data, opt, None, cam,
                        mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

glfw.terminate()
glfw.terminate()


fig,axs = plt.subplots(3)

axs[0].plot(contact_x)
axs[0].plot(obs_x)
axs[0].set_title("X-Contact Forces")

axs[1].plot(contact_y)
axs[1].plot(obs_y)
axs[1].set_title("Y-Contact Forces")

axs[2].plot(contact_z)
axs[2].plot(obs_z)
axs[2].set_title("Z-Contact Forces")
plt.subplots_adjust(hspace=0.5)
plt.legend("Actual", "Predicted")

# figs,con = plt.subplots(3)
# con[0].plot(obs_x)
# con[0].set_title("Momentum Observer X-contact")

# con[1].plot(obs_y)
# con[1].set_title("Momentum Observer Y-contact")
# con[2].plot(obs_z)
# con[2].set_title("Momentum Observer Z-contact")
# plt.subplots_adjust(hspace=0.5)
plt.show()