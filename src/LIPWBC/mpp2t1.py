import pinocchio as pin
import hppfcl
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import numpy as np
from math import cos, sin
from numpy import double
import time
from numpy.linalg import pinv
from scipy.linalg import block_diag
from qpsolvers import solve_qp
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
import matplotlib.pylab as plt; plt.ion()
import mujoco
import mujoco.viewer
from taskcls import TaskListc
from mpp2tfcls1 import ContactSchedule

modelfilename = 'test4.xml'
model = pin.buildModelFromMJCF(modelfilename,root_joint=pin.JointModelFreeFlyer())
collision_model = pin.buildGeomFromMJCF(model,modelfilename,pin.GeometryType.COLLISION)
visual_model = pin.buildGeomFromMJCF(model,modelfilename,pin.GeometryType.COLLISION)
data = model.createData()
collision_data = collision_model.createData()
visual_data = visual_model.createData()
model.frames[9].parentJoint = 4
model.frames[10].parentJoint = 4
model.frames[11].parentJoint = 4
model.frames[12].parentJoint = 4
model.frames[19].parentJoint = 7
model.frames[20].parentJoint = 7
model.frames[21].parentJoint = 7
model.frames[22].parentJoint = 7

m = mujoco.MjModel.from_xml_path(modelfilename)
d = mujoco.MjData(m)
x = np.zeros(model.nv+3*8+model.nv-6)
mrecord1 = []
mrecord2 = []
mrecord3 = []
mrecord4 = []
mrecord5 = []
mrecord6 = []
mrecord7 = []
mrecord8 = []
sinamp = 0.1
clist = ['t1','t2','t3','t4','t5','t6','t7','t8'] #如果是全部接触，应该用tuple
rcl = ['t1','t2','t3','t4']
lcl = ['t5','t6','t7','t8']
t1 = TaskListc(model,data)
t1.setCPNL(t1.fclist)
fs = 0
t1.ret()
t1.sm1(0.045,0)
jlist = [9,15,18,19]#, 6,12]
qobj = [0.4,0.4,0,0]#, -0.2,-0.2]
t1.setJST(jlist,qobj)
t1.jW = np.diag([10]*t1.jN)
landingtime = 0
cs1 = ContactSchedule()

def mycontroller(t1,d,ppos):
    # fs = t1.fs
    # t1.setCPNL(clist)
    t1.setStates(ppos,d.qvel)
    global fs
    global landingtime
    contactstate = 0
    contactstate2 = 0
    scst = [0,0]
    if d.sensordata[0]<0.01 and d.sensordata[1]<0.01 and d.sensordata[2]<0.01 and d.sensordata[3]<0.01:
        contactstate +=1
        scst[0] = 1
        t1.trl[0] = 0
    else:
        t1.trl[0] = 1
    if d.sensordata[4]<0.01 and d.sensordata[5]<0.01 and d.sensordata[6]<0.01 and d.sensordata[7]<0.01:
        contactstate +=1
        scst[1] = 1
        t1.trl[1] = 0
    else:
        t1.trl[1] = 1
    if all(value > 0.01 for value in d.sensordata[:4]):
        contactstate2 += 1
    if all(value > 0.01 for value in d.sensordata[4:8]):
        contactstate2 += 1

    angmomm = np.zeros((3,m.nv))
    mujoco.mj_angmomMat(m,d,angmomm,0)
    mhh = angmomm@d.qvel

    # if d.time >= 1.0 and fs == 20:
    #     t1.ret()
    #     jlist = [18,19]
    #     qobj = [0,0]
    #     t1.setJST(jlist,qobj)
    #     t1.jW = np.diag([1]*t1.jN)
    #     t1.fr_flag = 0
    #     # t1.jzs_flag = 1
    #     # t1.stablize_contact_flag = 1
    #     t1.angular_momenta_flag = 1
    #     t1.com_torque_task_flag = 0
    #     t1.hg_obj = np.array([0,0.0,0])
    #     t1.com_task_flag = 1
    #     t1.com_force_task_flag = 0
    #     t1.com_task_W = np.diag([10,10,10])
    #     t1.com_task_qobj = np.array([0.065,0,0.7+0.1*np.sin(6*(d.time - 1.5))])
    #     t1.com_task_vobj = np.array([0,0,0.6*np.cos(6*(d.time - 1.5))])
    if d.time >= 1.0 and fs == 0:
        rf1 = np.array([0.06,-0.15,0])
        lf1 = np.array([0.06,0.15,0])
        cs1.pushDST(0.4,np.array([0.0,0.0,0]))
        cs1.pushDS2SSOnL(rf1,lf1)
        cs1.printInfo()
        com,vcom = t1.getCoM()
        cx = np.array([com[0],vcom[0]])
        cy = np.array([com[1],vcom[1]])
        cs1.getTraj(d.time-1,cx,cy)
        t1.ret()
        jlist = [18,19]#,9,15,13] # wrist can do better
        qobj = [0,0]#,0.4,0.4,0.2]
        t1.setJST(jlist,qobj)
        t1.jW = np.diag([0.01]*2)#+[1]*3)
        # COM
        t1.com_task_flag = 0
        t1.com_force_task_flag = 1
        t1.com_task_W = np.diag([10,10,10]) #50,10,10
        t1.com_task_qobj = np.array([0.115,0.0,0.6]) #0.5/0.4
        t1.com_task_vobj = np.array([0,0,0])
        t1.com_vonly = False
        t1.Wclf = np.diag([0.1,0.1,0.1]) #0.5
        # 角动量
        t1.angular_momenta_flag = 0
        t1.com_torque_task_flag = 1
        t1.hgW = np.diag([5]*3) #15
        t1.kdhg = 65
        t1.hg_obj = np.array([-0.0,0.0,0]) #0.6
        t1.Wcrt = np.diag([5]*3) #15
        # 上身直立
        t1.fr_flag = 1
        t1.fr_W = np.diag([5]*3) #5;15
        # 节能
        t1.es_flag = 1
        t1.Wtau = np.diag([0.001]*(t1.nv-6)) #0.0004
        # 不动
        t1.jzs_flag = 0
        t1.jzs_W = 0.01*np.identity(t1.nv-6)
        # contact stab
        t1.stablize_contact_flag = 0
        t1.Wsc = 1*np.identity(8)
        t1.rc_flag = 0
        ###########################################
        print('time:',d.time,'fs_state_transition:0->1')
        fs = 1
    
    if fs == 0:
        # t1.setCPNL(t1.fclist)
        pass
    elif fs == 1:
        # t1.setCPNL(t1.fclist)
        ind = int((d.time-1)/0.01)
        if ind > 199:
            ind = 199
        t1.com_task_qobj = np.array([cs1.cx[0,ind],cs1.cy[0,ind],1]) #0.5/0.4
        t1.com_task_vobj = np.array([cs1.cx[1,ind],cs1.cy[1,ind],0])
        # print(t1.com_task_qobj)
        t1.setCPNL(cs1.sch[cs1.getCurrentsid(d.time-1)][1])
        pass
    elif fs == 2:
        # t1.setCPNL(t1.fclist)
        pass
    elif fs == 3:
        pass
    else:
        pass
        # raise ValueError('fs has bad value.')
    # t1.setComTaskObjs(mobj,mvobj)
    tclist = []
    for i in range(8): #这里是8！用cN就开始一直是空的了
        if t1.concou[i] > 0.01 :#and contactvelocity[3*i+2] < 0:
            tclist.append(t1.fclist[i])
    # t1.setCPNL(clist)
    t1.concou = np.zeros(len(t1.fclist))
    # t1.setCPNL(clist)
    t1.tsid()
    d.ctrl[:] = t1.tau

def set_cam_viewer(m, viewer):
    # 设置镜头跟随
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
    viewer.cam.distance = 3.0# 设置摄像机距离 
    viewer.cam.elevation = -20# 设置摄像机仰角
    
    # 设置摄像机为固定点模式
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    # 设定摄像机的位置和朝向
    # viewer.cam.lookat = [0, 0, 0]  # 摄像机视线的目标点
    # viewer.cam.distance = 5.0      # 摄像机与目标点的距离
    # viewer.cam.elevation = -20     # 摄像机仰角
    # viewer.cam.azimuth = 45        # 摄像机方位角
with mujoco.viewer.launch_passive(m, d) as viewer:
  # set_cam_viewer(m, viewer)
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  ii = 0
  mujoco.mj_forward(m, d)
  while viewer.is_running() and time.time() - start < 15:
    step_start = time.time()
    ppos = d.qpos.copy() # 没有copy就是会修改位置乱动
    quatl = ppos[3]
    ppos[3] = ppos[4]
    ppos[4] = ppos[5]
    ppos[5] = ppos[6]
    ppos[6] = quatl
    for i in range(8):
        t1.concou[i] += d.sensordata[i]

    if ii % 5 == 0:
        mycontroller(t1,d,ppos)
    ii += 1
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)
    mrecord1.append(d.subtree_com[1][0])
    mrecord2.append(d.subtree_com[1][2])
    mrecord3.append(d.subtree_com[1][1])
    mrecord4.append(sinamp*np.sin(d.time - 2.5)) #这里也要跟着改
    angmomm = np.zeros((3,m.nv))
    mujoco.mj_angmomMat(m,d,angmomm,0)
    hhere = angmomm@d.qvel
    mrecord5.append(hhere[0])
    mrecord6.append(hhere[1])
    mrecord7.append(hhere[2])
    mrecord8.append(d.qpos[12])

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

endp = 8000
plt.figure(1)
plt.subplot(311)
plt.plot(mrecord1[:endp])
plt.subplot(312)
plt.plot(mrecord3[:endp])
# plt.plot(mrecord4)
plt.subplot(313)
plt.plot(mrecord2[:endp])
plt.savefig('images/mpp2t1/com_traj1_mpp2t1.png')