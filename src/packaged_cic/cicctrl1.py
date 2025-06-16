import pinocchio as pin
import hppfcl
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import numpy as np
from math import cos, sin
from numpy import double
import time
from numpy.linalg import pinv
from scipy.linalg import block_diag, solve
from qpsolvers import solve_qp
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
import matplotlib.pylab as plt; plt.ion()
import mujoco
import mujoco.viewer

class cic1:
    def __init__(self,modelfilename):#model,data):
        self.N = 90
        self.dt = 0.01
        # self.model = model
        # self.data = data
        self.model = pin.buildModelFromMJCF(modelfilename,root_joint=pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.model.frames[9].parentJoint = 4
        self.model.frames[10].parentJoint = 4
        self.model.frames[11].parentJoint = 4
        self.model.frames[12].parentJoint = 4
        self.model.frames[19].parentJoint = 7
        self.model.frames[20].parentJoint = 7
        self.model.frames[21].parentJoint = 7
        self.model.frames[22].parentJoint = 7
        self.model.frames[self.model.getFrameId('t1')].parentFrame = self.model.getFrameId('link3')
        self.model.frames[self.model.getFrameId('t2')].parentFrame = self.model.getFrameId('link3')
        self.model.frames[self.model.getFrameId('t3')].parentFrame = self.model.getFrameId('link3')
        self.model.frames[self.model.getFrameId('t4')].parentFrame = self.model.getFrameId('link3')
        self.model.frames[self.model.getFrameId('t5')].parentFrame = self.model.getFrameId('link6')
        self.model.frames[self.model.getFrameId('t6')].parentFrame = self.model.getFrameId('link6')
        self.model.frames[self.model.getFrameId('t7')].parentFrame = self.model.getFrameId('link6')
        self.model.frames[self.model.getFrameId('t8')].parentFrame = self.model.getFrameId('link6')
        self.q = pin.neutral(self.model).copy()
        self.q[2] = 1.10
        self.qdot = np.zeros(self.model.nv)
        self.fclist= ['t1','t2','t3','t4','t5','t6','t7','t8']
        self.clist = []
        self.fcN = len(self.fclist)
        self.cN = len(self.clist)
        self.qhist = [self.q]
        self.fhist = []
        self.aqfree = np.zeros(self.model.nv)
        self.M = None
        self.c = None
        self.com = None
        self.Jcom = None
        self.vcom = None
        self.Ag = None
        self.vcom2 = None
        self.vcom3 = None
        self.vcom4 = None
        self.com_task_qobj = None
        self.contactposition = np.zeros(3*self.cN)
        self.contactvelocity = np.zeros(3*self.cN)
        self.Jdv = np.zeros(3*self.cN)
        self.Jdvc = np.zeros(3*self.cN)
        self.o_Jtool = np.zeros((3*self.cN,self.model.nv))
        self.eR = 0.01
        self.Gde = None
        self.tau = None
        self.ttau = np.zeros(self.model.nv-6)

        self.Wtau = np.diag([0.018]*(self.model.nv-6))
        self.jW = None
        self.jWe = 10
        self.jWdiag = None
        self.jlist = None
        self.jN = None
        self.qobj = None
        self.kpj = 100
        self.kdj = 10
        self.ajW = None
        self.ajWe = 10
        self.ajlist = None
        self.ajN = None
        self.aqobj = None

        self.rfpos = None
        self.lfpos = None
        self.rhip = None
        self.lhip = None
        self.fRT = None
        self.fpi = None
        self.fp = None
        
        self.mrec = [[] for _ in range(45)]
        self.fs = 0
        self.concou = np.zeros(8)

    def getDK(self):
        model = self.model
        data = self.data
        self.M = pin.crba(model, data, self.q)
        pin.computeMinverse(model,data,self.q)
        self.c = pin.nle(model, data, self.q, self.qdot)
        pin.forwardKinematics(model,data,self.q,self.qdot,np.zeros(model.nv))
        pin.updateFramePlacements(model, data)
        com = pin.centerOfMass(model,data,self.q,self.qdot,False)
        self.com = com
        self.Jcom = pin.jacobianCenterOfMass(model,data,self.q,False)
        self.vcom = self.Jcom@self.qdot
        pin.computeCentroidalMap(model,data,self.q)
        self.Ag = data.Ag
        self.vcom2 = self.Ag[:3,:]@self.qdot
        self.vcom3 = data.vcom[0]
        dt = 0.001
        qnext = pin.integrate(model,self.q,self.qdot*dt)
        comnext = pin.centerOfMass(model,data,qnext,False)
        self.vcom4 = (comnext - com)/dt
        self.com = pin.centerOfMass(model,data,self.q,self.qdot,False)
        rfid = model.getFrameId('link3')
        lfid = model.getFrameId('link6')
        rhid = model.getFrameId('Composite_HIPRY') #Composite_HIPRY
        lhid = model.getFrameId('Composite_HIPLY') #Composite_HIPLY
        self.rfpos = data.oMf[rfid].translation
        self.lfpos = data.oMf[lfid].translation
        self.rhip = data.oMf[rhid].translation
        self.lhip = data.oMf[lhid].translation
        vecr2l = self.lhip - self.rhip
        vecr2lxynorm = np.sqrt(vecr2l[0]*vecr2l[0] + vecr2l[1]*vecr2l[1])
        fRy = np.array([vecr2l[0]/vecr2lxynorm,vecr2l[1]/vecr2lxynorm,0])
        fRz = np.array([0,0,1])
        fRx = np.cross(fRy,fRz)
        self.fRT = np.array([fRx,fRy,fRz])
        self.fp = vecr2l/2 + self.rhip
        self.fpi = -self.fRT@self.fp
        
    def getCon(self):
        model = self.model
        data = self.data
        # self.clist = []
        # fcontactposition = np.zeros(3*self.fcN)
        # for i in range(self.fcN):
        #     fcontactposition[3*i:3*(i+1)] = data.oMf[model.getFrameId(self.fclist[i])].translation
        #     if fcontactposition[3*i+2] < 0:
        #         self.clist.append(self.fclist[i])
        self.cN = len(self.clist)
        self.contactposition = np.zeros(3*self.cN)
        for i in range(self.cN):
            self.contactposition[3*i:3*(i+1)] = data.oMf[model.getFrameId(self.clist[i])].translation
        self.contactvelocity = np.zeros(3*self.cN)
        for i in range(self.cN):
            self.contactvelocity[3*i:3*(i+1)] = pin.getFrameVelocity(model, data, model.getFrameId(self.clist[i]), pin.LOCAL_WORLD_ALIGNED).linear
        self.Jdv = np.zeros(3*self.cN)
        self.Jdvc = np.zeros(3*self.cN)
        for i in range(self.cN):
            self.Jdv[3*i:3*(i+1)] = pin.getFrameAcceleration(model,data,model.getFrameId(self.clist[i]),pin.LOCAL_WORLD_ALIGNED).linear
            self.Jdvc[3*i:3*(i+1)] = pin.getFrameClassicalAcceleration(model, data, model.getFrameId(self.clist[i]), pin.LOCAL_WORLD_ALIGNED).linear
        self.o_Jtool = np.zeros((3*self.cN,model.nv))
        for i in range(self.cN):
            self.o_Jtool[3*i:3*(i+1),:] = pin.computeFrameJacobian(model,data,self.q,model.getFrameId(self.clist[i]), pin.LOCAL_WORLD_ALIGNED)[:3,:]
        # eR = 0.00001
        self.Gde = self.o_Jtool@data.Minv@self.o_Jtool.T + self.eR*np.identity(3*self.cN)

    def tsid(self):
        model = self.model
        data = self.data
        # self.getDK()
        self.getCon()
        if self.jlist is None:
            self.jN = 0
        else:
            self.jN = len(self.jlist)
        self.jW = np.diag([self.jWe]*self.jN)
        if self.jWdiag:
            self.jW = np.diag(self.jWdiag)
        # ratoajw = 1
        # if self.jN>10:
        #     self.jW[4][4]=ratoajw*self.jWe
        #     self.jW[5][5]=ratoajw*self.jWe
        #     self.jW[10][10]=ratoajw*self.jWe
        #     self.jW[11][11]=ratoajw*self.jWe
        if self.ajlist is None:
            self.ajN = 0
        else:
            self.ajN = len(self.ajlist)
        self.ajW = np.diag([self.ajWe]*self.ajN)
        while True:
            self.getCon()
            Pqdd = np.zeros((model.nv+3*self.cN,model.nv+3*self.cN))
            qqdd = np.zeros(model.nv+3*self.cN)
            Jes = np.block([self.M[6:,:],-self.o_Jtool.T[6:,:]])
            esbias = -self.c[6:]
            Pqdd += Jes.T@self.Wtau@Jes
            qqdd += -Jes.T@self.Wtau@esbias
            if self.jN != 0:
                Jj = np.zeros((self.jN,model.nv+3*self.cN))
                for j in range(self.jN):
                    Jj[j,self.jlist[j]] = 1
                qhere = np.array([self.q[i+1] for i in self.jlist])
                qdhere = np.array([self.qdot[i] for i in self.jlist])
                task_ja = -self.kpj*(qhere - self.qobj) - self.kdj*qdhere
                # task_jv = qdhere + task_ja*self.dt
                Pqdd += Jj.T@self.jW@Jj
                qqdd += -(Jj.T)@self.jW@task_ja
            if self.ajN != 0:
                aJj = np.zeros((self.ajN,model.nv+3*self.cN))
                for j in range(self.ajN):
                    aJj[j,self.ajlist[j]] = 1
                task_aja = self.aqobj
                Pqdd += aJj.T@self.ajW@aJj
                qqdd += -(aJj.T)@self.ajW@task_aja
            
            QPP  = Pqdd + 0.001*np.identity(model.nv+3*self.cN)
            QPq = qqdd
            A = np.block([[self.M[:6,:],-self.o_Jtool.T[:6,:]],[self.o_Jtool,0.001*np.identity(3*self.cN)]])
            A[:6,:6] += 0.00001*np.identity(6)
            b = np.concatenate((-self.c[:6],-self.Jdv- 5*(self.contactvelocity)))#- 20*(contactposition - origincontactposition) - 5*(contactvelocity)
            maxset = 100000
            lb = np.array([-maxset]*6+[-15000]*14+[-19000,-19000,-19000]*(self.cN)) #10.1
            ub = np.array([ maxset]*6+[ 15000]*14+[ 19000]*(3*self.cN))
            # self.x = solve_qp(QPP,QPq,A=A,b=b,lb=lb,ub=ub,solver="proxqp").copy()
            self.x = proxqp_solve_qp(QPP,QPq,A=A,b=b,lb=lb,ub=ub,eps_abs=1e-7)
            liftoffstate = 0
            herelist = self.clist.copy()
            for i in range(self.cN):
                if self.clist  and self.x[model.nv+2+3*i] <= -0.00001: #投影到法向
                    herelist.remove(self.clist[i])
                    liftoffstate += 1
            self.clist = herelist
            if liftoffstate == 0:
                break
        self.tau = self.M[6:,:]@self.x[:model.nv] + (self.c[6:]) - self.o_Jtool.T[6:,:]@self.x[model.nv:]
        
    def import_model(self,modelfilename):
        self.model = pin.buildModelFromMJCF(modelfilename,root_joint=pin.JointModelFreeFlyer())
        model = self.model
        collision_model = pin.buildGeomFromMJCF(model,modelfilename,pin.GeometryType.COLLISION)
        visual_model = pin.buildGeomFromMJCF(model,modelfilename,pin.GeometryType.COLLISION)
        self.data = model.createData()
        collision_data = collision_model.createData()
        visual_data = visual_model.createData()
        # model.frames[9].parentJoint = 4
        # model.frames[10].parentJoint = 4
        # model.frames[11].parentJoint = 4
        # model.frames[12].parentJoint = 4
        # model.frames[19].parentJoint = 7
        # model.frames[20].parentJoint = 7
        # model.frames[21].parentJoint = 7
        # model.frames[22].parentJoint = 7
        # model.frames[model.getFrameId('t1')].parentFrame = model.getFrameId('link3')
        # model.frames[model.getFrameId('t2')].parentFrame = model.getFrameId('link3')
        # model.frames[model.getFrameId('t3')].parentFrame = model.getFrameId('link3')
        # model.frames[model.getFrameId('t4')].parentFrame = model.getFrameId('link3')
        # model.frames[model.getFrameId('t5')].parentFrame = model.getFrameId('link6')
        # model.frames[model.getFrameId('t6')].parentFrame = model.getFrameId('link6')
        # model.frames[model.getFrameId('t7')].parentFrame = model.getFrameId('link6')
        # model.frames[model.getFrameId('t8')].parentFrame = model.getFrameId('link6')
        
def mycontroller(c1,d,ppos):
    c1.q = ppos
    c1.qdot = d.qvel
    # fs = c1.fs
    # concou = c1.concou
    global starttime
    contactstate = 0
    contactstate2 = 0
    rls = 0
    lls = 0
    scst = [0,0]
    c1.getDK()
    h = 1
    g = 9.8
    omega = np.sqrt(g/h)
    mujoco.mj_subtreeVel(m,d)
    c1.vcom = d.subtree_linvel[1]
    eta = c1.com + c1.vcom/omega
    # feta = c1.fRT@eta + c1.fpi
    frfpos = c1.fRT@c1.rfpos + c1.fpi
    flfpos = c1.fRT@c1.lfpos + c1.fpi
    fcom = c1.fRT@c1.com + c1.fpi
    fvcom = c1.fRT@c1.vcom
    # print(c1.vcom)
    # print('fvcom:',fvcom)
    fvp = c1.fRT@(c1.vcom/omega)
    feta = fcom + fvp
    c1.mrec[9].append(eta[1])
    c1.mrec[8].append(c1.vcom[1])
    c1.mrec[10].append(c1.rfpos[0])
    c1.mrec[11].append(c1.lfpos[0])
    c1.mrec[12].append(eta[0])
    c1.mrec[13].append(c1.fpi[1])
    c1.mrec[14].append(feta[1])
    c1.mrec[15].append(feta[0])
    c1.mrec[16].append(frfpos[1])
    c1.mrec[17].append(flfpos[1])
    c1.mrec[18].append(fcom[1])
    c1.mrec[19].append(fcom[2])
    c1.mrec[20].append(fvcom[0])
    c1.mrec[21].append(fvp[0])
    c1.mrec[22].append(c1.vcom[0])
    c1.mrec[27].append(fvcom[1])
    c1.mrec[28].append(c1.com[0])
    c1.mrec[29].append(c1.com[1])
    c1.mrec[33].append(c1.vcom[0])
    c1.mrec[34].append(c1.vcom2[0])
    c1.mrec[35].append(c1.vcom3[0])
    c1.mrec[36].append(c1.vcom4[0])
    if d.sensordata[0]<0.01 and d.sensordata[1]<0.01 and d.sensordata[2]<0.01 and d.sensordata[3]<0.01:
        contactstate +=1
        rls = 1
    #     scst[0] = 1
    #     c1.trl[0] = 0
    # else:
    #     c1.trl[0] = 1
    if d.sensordata[4]<0.01 and d.sensordata[5]<0.01 and d.sensordata[6]<0.01 and d.sensordata[7]<0.01:
        contactstate +=1
        lls = 1
    #     scst[1] = 1
    #     c1.trl[1] = 0
    # else:
    #     c1.trl[1] = 1
    if all(value > 0.01 for value in d.sensordata[:4]):
        contactstate2 += 1
    if all(value > 0.01 for value in d.sensordata[4:8]):
        contactstate2 += 1

    if feta[1]-frfpos[1] > 0.19 and c1.fs == 0: #eta[1] > 0.08 #0.07
        # print(c1.vcom)
        # print('fvcom:',fvcom)
        # print(c1.com)
        # print('fcom:',fcom)
        # print(fvp)
        # print(c1.rfpos)
        # print(c1.lfpos)
        # print(c1.fRT@c1.rfpos+c1.fpi)
        # print(c1.fRT@c1.lfpos+c1.fpi)
        pass
        ###########################################
        print('time:',d.time,'fs_state_transition:0->1')
        c1.fs = 1
    if flfpos[1]-feta[1] > 0.19 and c1.fs == 1: #eta[1] < -0.08
        pass
        ###########################################
        print('time:',d.time,'fs_state_transition:1->0')
        c1.fs = 0

    #knee:0.7 ank:0.12 
    output = 0
    hipyor = 0
    hipyol = 0
    apitr = 0
    apitl = 0
    kneer = 0
    kneel = 0
    hipzr = 0
    hipzl = 0
    kneeliftangle = 0.7 #0.7 #1.5 0.7
    amoo = 0.14 #0.08
    swingang = 0.1
    anklebias = 0.05 #0.08 $0.16 #0.06
    pos2ang = 0.9
    posbias = -0.0 #0也行
    hipb = 0.0 #0.1 0.12
    liftratio = 0
    b = 0.29 #0.2
    # b = 0.2*sin(d.time*2*np.pi/6)+0.25
    # if d.time <4: #4
    #     b = 0.1
    # else:
    #     b = 0.4 #0.4
    c1.mrec[23].append(b)
    if c1.fs == 0:
        # t1.setCPNL(t1.fclist)
        output = -amoo
        kneel = kneeliftangle
        diffpos = fcom[0] + b*fvcom[0] - frfpos[0]#feta[0] - frfpos[0] + posbias
        ang = diffpos#diffpos*pos2ang
        kneel = ang*0 + kneeliftangle
        if lls == 1:
            if abs(diffpos) >= 2:
                print('error!!!!!!')
            ang = np.arcsin(diffpos/2)
            # ang = swingang
            hipyor = ang
            hipyol = -ang
            # hipzr = -0.5
        hipyol += -kneeliftangle/2+hipb+ang*liftratio
        apitl += -anklebias-kneeliftangle/2-ang*liftratio-hipb
        apitr += -anklebias 
        if fvcom[1] > 0:
            kneel = 0
            hipyol += kneeliftangle/2-hipb-ang*liftratio
            apitl += kneeliftangle/2+ang*liftratio+hipb
        pass
    elif c1.fs == 1:
        # t1.setCPNL(t1.fclist)
        output = amoo
        kneer = kneeliftangle
        diffpos = fcom[0] + b*fvcom[0] - flfpos[0]#feta[0] - flfpos[0] + posbias
        ang = diffpos#diffpos*pos2ang
        kneer = ang*0 + kneeliftangle
        if rls == 1:
            if abs(diffpos) >= 2:
                print('error!!!!!!')
            ang = np.arcsin(diffpos/2)
            # ang = swingang
            hipyor = -ang
            hipyol = ang
            # hipzl = 0.5
        hipyor += -kneeliftangle/2+hipb+ang*liftratio
        apitl += -anklebias
        apitr += -anklebias-kneeliftangle/2-ang*liftratio-hipb
        if fvcom[1] < 0:
            kneer = 0
            hipyor += kneeliftangle/2-hipb-ang*liftratio
            apitr += kneeliftangle/2+ang*liftratio+hipb
        pass
    else:
        pass
        # raise ValueError('c1.fs has bad value.')
    
    tclist = []
    for i in range(8): #这里是8！用cN就开始一直是空的了
        if c1.concou[i] > 0.01 :#and contactvelocity[3*i+2] < 0:
            tclist.append(c1.fclist[i])
    c1.clist = tclist
    c1.concou = np.zeros(len(c1.fclist))
    c1.jlist = [6,7,8,9,10,11,              12,13,14,15,16,17,          18,19]
    c1.qobj =  [hipyor,output,hipzr,kneer,apitr,0,   hipyol,output,hipzl,kneel,apitl,0,   0.0,0]
    c1.Wtau[5,5] = 0.1
    c1.Wtau[11,11] = 0.1
    c1.tsid()
    # d.ctrl[:] = c1.tau
    # c1.mrec[24].append(c1.tau[0])
    # c1.mrec[25].append(c1.tau[3])
    # c1.mrec[26].append(c1.tau[4])

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

if __name__ == "__main__":
    modelfilename = 'test4.xml'
    m = mujoco.MjModel.from_xml_path(modelfilename)
    d = mujoco.MjData(m)
    sinamp = 0.1
    # c1 = cic1(model,data)
    c1 = cic1(modelfilename)

    with mujoco.viewer.launch_passive(m, d) as viewer:
    # set_cam_viewer(m, viewer)
    # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        ii = 0
        qst = pin.Quaternion(pin.AngleAxis(0,np.array([0,0,1])).matrix()).coeffs().copy()
        qslt = qst[3]
        qst[3] = qst[2]
        qst[2] = qst[1]
        qst[1] = qst[0]
        qst[0] = qslt
        d.qpos[3:7] = qst
        mujoco.mj_forward(m, d)
        while viewer.is_running() and time.time() - start < 25:
            step_start = time.time()
            ppos = d.qpos.copy() # 没有copy就是会修改位置乱动
            quatl = ppos[3]
            ppos[3] = ppos[4]
            ppos[4] = ppos[5]
            ppos[5] = ppos[6]
            ppos[6] = quatl
            for i in range(8):
                c1.concou[i] += d.sensordata[i]

            if ii % 5 == 0:
                mycontroller(c1,d,ppos)
            ii += 1
            # c1.ttau -= 0.1*(c1.ttau-c1.tau)
            # uot = 120
            # lot = -120
            # for i in range(14):
            #     if c1.ttau[i] > uot:
            #         c1.ttau[i] = uot
            #     elif c1.ttau[i] < lot:
            #         c1.ttau[i] = lot
            c1.ttau = c1.tau
            d.ctrl[:] = c1.ttau
            c1.mrec[24].append(c1.ttau[0])
            c1.mrec[25].append(c1.ttau[3])
            c1.mrec[26].append(c1.ttau[4])
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics. 
            mujoco.mj_step(m, d)
            c1.mrec[1].append(d.subtree_com[1][0])
            c1.mrec[2].append(d.subtree_com[1][2])
            c1.mrec[3].append(d.subtree_com[1][1])
            c1.mrec[4].append(sinamp*np.sin(d.time - 2.5)) #这里也要跟着改
            angmomm = np.zeros((3,m.nv))
            mujoco.mj_angmomMat(m,d,angmomm,0)
            hhere = angmomm@d.qvel
            c1.mrec[5].append(hhere[0])
            c1.mrec[6].append(hhere[1])
            c1.mrec[7].append(hhere[2])
            # c1.mrecord8.append(d.qpos[8])
            # c1.mrecord9.append(d.qpos[14])
            mujoco.mj_subtreeVel(m,d)
            c1.mrec[30].append(d.subtree_linvel[1][0])
            c1.mrec[31].append(d.subtree_linvel[1][1])
            c1.mrec[32].append(d.subtree_linvel[1][2])

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)