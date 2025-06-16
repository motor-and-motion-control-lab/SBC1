import pinocchio as pin
import hppfcl
import numpy as np
from math import cos, sin
from numpy import double
import time
from numpy.linalg import pinv
from scipy.linalg import block_diag, solve
from qpsolvers import solve_qp
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
import matplotlib.pylab as plt; plt.ion()
# import mujoco
# import mujoco.viewer
# from rich import print as rprint

class cic1:
    def __init__(self,modelfilename_pin):
        self.N = 90
        self.dt = 0.01
        self.model = pin.buildModelFromMJCF(modelfilename_pin,root_joint=pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.q = pin.neutral(self.model).copy()
        self.q[2] = 1.10
        self.qdot = np.zeros(self.model.nv)
        self.fclist= ['{}_sensor_{}{}'.format('l', i, j) for i in range(1, 4) for j in range(1, 5)] + ['{}_sensor_{}{}'.format('r', i, j) for i in range(1, 4) for j in range(1, 5)]
        self.clist = []
        self.fcN = len(self.fclist)
        self.cN = len(self.clist)
        self.qhist = [self.q]
        self.fhist = []
        self.aqfree = np.zeros(self.model.nv)
        self.M, self.c, self.com, self.Jcom, self.vcom = None, None, None, None, None
        self.com_task_qobj = None
        self.contactposition = np.zeros(3*self.cN)
        self.contactvelocity = np.zeros(3*self.cN)
        self.Jdv = np.zeros(3*self.cN)
        self.Jdvc = np.zeros(3*self.cN)
        self.o_Jtool = np.zeros((3*self.cN,self.model.nv))
        self.eR = 0.01
        self.Gde = None
        self.tau = None

        self.Wtau = np.diag([0.008]*(self.model.nv-6))
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

        self.rfpos, self.lfpos, self.rhip, self.lhip, self.fRT, self.fpi = None, None, None, None, None, None
        
        self.mrec = [[] for _ in range(45)]
        self.fs = 0
        self.concou = np.zeros(24)

    def getDK(self):
        model = self.model
        data = self.data
        self.M = pin.crba(model, data, self.q)
        pin.computeMinverse(model,data,self.q)
        self.c = pin.nle(model, data, self.q, self.qdot)
        pin.forwardKinematics(model,data,self.q,self.qdot,np.zeros(model.nv))
        pin.updateFramePlacements(model, data)
        com = pin.centerOfMass(model,data,self.q,False)
        self.com = com
        self.Jcom = pin.jacobianCenterOfMass(model,data,self.q,False)
        self.vcom = self.Jcom@self.qdot
        rfid = model.getFrameId('r_leg_ankle_pitch_Link')
        lfid = model.getFrameId('l_leg_ankle_pitch_Link')
        rhid = model.getFrameId('r_leg_hip_roll_Link')
        lhid = model.getFrameId('l_leg_hip_roll_Link')
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
        fp = vecr2l/2 + self.rhip
        self.fpi = -self.fRT@fp
        
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
            lb = np.array([-maxset]*6+[-15000]*12+[-19000,-19000,-19000]*(self.cN)) #10.1
            ub = np.array([ maxset]*6+[ 15000]*12+[ 19000]*(3*self.cN))
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
        
def mycontroller(c1,ppos,qvel,time=0):
    # global c1.fs, c1.concou
    rls, lls = 0, 0
    c1.getDK()
    h = 1
    g = 9.8
    omega = np.sqrt(g/h)
    eta = c1.com + c1.vcom/omega
    # feta = c1.fRT@eta + c1.fpi
    frfpos = c1.fRT@c1.rfpos + c1.fpi
    flfpos = c1.fRT@c1.lfpos + c1.fpi
    fcom = c1.fRT@c1.com + c1.fpi
    fvcom = c1.fRT@c1.vcom
    fvp = c1.fRT@(c1.vcom/omega)
    feta = fcom + fvp
    lls = int(np.all(c1.concou[:12] < 0.01))
    rls = int(np.all(c1.concou[12:] < 0.01))

    if c1.fs == 0 and feta[1] - frfpos[1] > 0.2:
        print(f'time: {time}, fs_state_transition: 0 -> 1')
        c1.fs = 1
    elif c1.fs == 1 and flfpos[1] - feta[1] > 0.2:
        print(f'time: {time}, fs_state_transition: 1 -> 0')
        c1.fs = 0

    output, hipyor, hipyol, apitr, apitl, kneer, kneel = 0, 0, 0, 0, 0, 0, 0
    
    # 可调参数
    kneeliftangle = 1.0 #0.7 #1.5
    amoo = 0.12 #0.08
    anklebias = 0.12 #0.08 $0.16 #0.06
    pos2ang = 0.9
    posbias = -0.0 #0也行
    b = 0.1
    # b = 0.15*sin(d.time*2*np.pi/6)+0.25
    # if d.time <3: #4
    #     b = 0.1
    # else:
    #     b = 0.3 #0.4
    if c1.fs == 0:
        output = -amoo
        kneel = kneeliftangle
        diffpos = fcom[0] + b*fvcom[0] - frfpos[0]#feta[0] - frfpos[0] + posbias
        ang = diffpos#diffpos*pos2ang
        if lls == 1:
            if abs(diffpos) >= 2*0.55:
                print('error!!!!!!')
            ang = np.arcsin(diffpos/2/0.55)
            hipyor = ang
            hipyol = -ang
        hipyol += -kneeliftangle/2
        apitl += -anklebias-kneeliftangle/2#-ang
        apitr += -anklebias 
        if fvcom[1] > 0:
            kneel = 0
            hipyol += kneeliftangle/2
            apitl += kneeliftangle/2
    elif c1.fs == 1:
        output = amoo
        kneer = kneeliftangle
        diffpos = fcom[0] + b*fvcom[0] - flfpos[0]#feta[0] - flfpos[0] + posbias
        ang = diffpos#diffpos*pos2ang
        if rls == 1:
            if abs(diffpos) >= 2*0.55:
                print('error!!!!!!')
            ang = np.arcsin(diffpos/2/0.55)
            hipyor = -ang
            hipyol = ang
        hipyor += -kneeliftangle/2
        apitl += -anklebias
        apitr += -anklebias-kneeliftangle/2#-ang
        if fvcom[1] < 0:
            kneer = 0
            hipyor += kneeliftangle/2
            apitr += kneeliftangle/2
    else:
        pass
        # raise ValueError('c1.fs has bad value.')
    c1.q = ppos
    c1.qdot = qvel
    c1.clist = [c1.fclist[i] for i in range(24) if c1.concou[i] > 0.01]
    c1.concou = np.zeros(len(c1.fclist))
    
    c1.jlist = [8,6,7,9,10,11,
                14,12,13,15,16,17]
    c1.qobj =  [hipyol,output,0,kneel,apitl,0,
                hipyor,output,0,kneer,apitr,0]
    # c1.Wtau[5,5] = 0.1
    # c1.Wtau[11,11] = 0.1
    c1.tsid()