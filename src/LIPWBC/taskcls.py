import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from math import cos, sin
from numpy.linalg import pinv
from scipy.linalg import block_diag
from qpsolvers import solve_qp
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
from numpy.linalg import pinv

class TaskListc:
    def __init__(self,model,data):
        self.model = model
        self.data = data
        self.nv = model.nv
        # self.dPqdd = np.zeros((nv,nv))
        # self.dqqdd = np.zeros(nv)
        self.dt = 0.01
        self.ii = 0
        
        self.cN = 0
        # self.Jc = None
        self.fclist = ('t1','t2','t3','t4','t5','t6','t7','t8')
        self.clts = np.ones(len(self.fclist))
        self.clist = []
        self.minfz = 0.15
        self.q = np.zeros(model.nq)
        self.qdot = np.zeros(model.nv)
        self.invR = None
        self.Wf = []
        
        self.com_task_flag = 0
        self.com_task_W = np.diag([10]*3)
        self.com_task_qobj = np.array([0.02,0,0.6])
        self.com_task_vobj = np.array([0,0,0])
        self.com_SM = np.identity(model.nv)
        self.com_vonly = False

        self.angular_momenta_flag = 0
        self.hgW = np.diag([10]*3)
        self.hg_obj = np.zeros(3)
        self.kdhg = 65

        self.Wclf = np.diag([0.5]*3)
        self.Wcrt = np.diag([5]*3)
        self.com_force_task_flag = 0
        self.com_torque_task_flag = 0

        self.pN = 0
        self.plist = []
        self.pobj = []
        self.pvobj = []
        self.pW = None
        self.pSM = np.identity(model.nv)

        self.jN = 0
        self.jlist = []
        self.qobj = []
        self.jW = None
        self.kpj = 100
        self.kdj = 20

        self.fr_flag = 0
        self.fr_obj = np.array([0,0,0,1]) #pin.Quaternion(0.7, 0.2, 0.2, 0.6).normalized()#
        self.fr_W = np.diag([0.5]*3) #50

        self.x = np.zeros(model.nv+3*self.cN+model.nv-6)
        self.tau = np.zeros(model.nv-6)
        self.cf = np.zeros(3*8)

        self.srleg = np.zeros((model.nv,model.nv))
        self.slleg = np.zeros((model.nv,model.nv))
        self.sfloat = np.zeros((model.nv,model.nv))
        self.rrleg = np.identity(model.nv)
        self.rlleg = np.identity(model.nv)
        self.rfloat = np.identity(model.nv)
        for i in range(6):
            self.srleg[i+6,i+6] = 1
            self.slleg[i+12,i+12] = 1
            self.sfloat[i,i] = 1
            self.rrleg[i+6,i+6] = 0
            self.rlleg[i+12,i+12] = 0
            self.rfloat[i,i] = 0
        
        self.Wreg = 0.001*np.identity(model.nv)
        for i in range(6):
            self.Wreg[i,i] = 0.0001
        self.bend_knee_flag = 0
        self.es_flag = 1
        self.Wtau = np.diag([0.018]*(model.nv-6))
        self.jzs_flag = 0
        self.jzs_W = 0.01*np.identity(self.nv-6)
        self.stablize_contact_flag = 0
        self.rcl = []
        self.Wsc = 0.1*np.identity(8)

        self.fs = 0
        self.concou = np.zeros(len(self.fclist))

        self.trl = [1,1]
        self.rc_flag = 0
        self.lc_flag = 0

        self.com = np.zeros(3)
        
    # def setCPNJ(self,cN):
    #     self.cN = cN
    #     self.Jc = np.zeros((3*cN,self.nv))
    def setCPNL(self,clist):
        self.cN = len(clist)
        self.clist = clist
        self.invR = np.diag([200]*3*self.cN) #1000就不行了，注意这里是实际的invR多乘以dt后的，所以很大了
        self.Wf = np.diag([10,10,10]*(self.cN))
        # self.concou = np.zeros(self.cN)
    def addComTask(self,W=[20]*3): #10,我记得这里原来10也行，很怪
        self.com_task_flag = 1
        self.com_task_W = np.diag(W)
    def setComTaskObjs(self,qobj,vobj=np.zeros(3)):
        self.com_task_qobj = qobj
        self.com_task_vobj = vobj
    def resetComTask(self):
        self.com_task_flag = 0
        self.com_task_W = np.zeros((3,3))
        self.com_task_qobj = np.array([0.02,0,0.6])
        self.com_task_vobj = np.array([0,0,0])
    def setTSTP(self,plist,pobj):
        self.pN = len(plist)
        self.plist = plist
        self.pobj = pobj
        self.pvobj = np.array([0,0,0]*self.pN)
        self.pW = np.diag([10,10,10]*self.pN)
    def setJST(self,jlist,qobj):
        self.jN = len(jlist)
        self.jlist = jlist
        self.qobj = qobj
        self.jW = np.diag([10]*self.jN)
    def addFRT(self):
        self.fr_flag = 1
    def removeFRT(self):
        self.fr_flag = 0
    def setFRT(self,fr_obj=np.array([0,0,0,1])):
        self.fr_obj = fr_obj
    def setStates(self,q,qdot):
        self.q = q
        self.qdot = qdot
    def getJcomdotqdot(self,q,qdot):
        dt = 0.0001
        qnext = pin.integrate(self.model,q,qdot*dt)
        jcomnow = pin.jacobianCenterOfMass(self.model,self.data,q,False)
        jcomnext = pin.jacobianCenterOfMass(self.model,self.data,qnext,False)
        return (jcomnext@qdot - jcomnow@qdot)/dt
    def clamp(self,q,l,u):
        if q < l:
            q = l
        if q > u:
            q = u
    def sm1(self,x,y):
        # COM
        self.com_task_flag = 1
        self.com_task_W = np.diag([10,10,0])
        self.setComTaskObjs(np.array([x,y,0.6]))
        self.com_vonly = False
        self.Wclf = np.diag([5,5,0]) #0.5
        # 角动量
        self.angular_momenta_flag = 0
        self.hgW = np.diag([10]*3)
        self.kdhg = 65
        self.hg_obj = np.zeros(3) 
        self.Wcrt = np.diag([5]*3)
        # 上身直立
        self.fr_flag = 1
        self.fr_W = np.diag([5]*3)
        # 节能
        self.es_flag = 1
        self.Wtau = np.diag([0.018]*(self.nv-6))
        # 不动
        self.jzs_flag = 0
        self.jzs_W = 0.01*np.identity(self.nv-6)
        # contact stab
        self.stablize_contact_flag = 0
    def ret(self):
        self.es_flag = 0
        self.com_force_task_flag = 0
        self.com_torque_task_flag = 0
        self.com_task_flag = 0
        self.angular_momenta_flag = 0
        self.fr_flag = 0
        self.pN = 0
        self.jN = 0
        self.jzs_flag = 0
        self.stablize_contact_flag = 0
    def setrecl(self):
        # fl = set(self.fclist.copy())
        # cl = set(self.clist.copy())
        # wfcl = lsit(fl.difference(cl))
        pass
    def getJcomdotqdot(self,q,qdot):
        dt = 0.0001
        qnext = pin.integrate(self.model,q,qdot*dt)
        jcomnow = pin.jacobianCenterOfMass(self.model,self.data,q,False)
        jcomnext = pin.jacobianCenterOfMass(self.model,self.data,qnext,False)
        return (jcomnext@qdot - jcomnow@qdot)/dt
    def getJhdotqdot(self,q,qdot):
        dt = 0.0001
        qnext = pin.integrate(self.model,q,qdot*dt)
        pin.computeCentroidalMap(self.model,self.data,q)
        jhnow = self.data.Ag[3:,:]
        pin.computeCentroidalMap(self.model,self.data,qnext)
        jhnext = self.data.Ag[3:,:]
        return (jhnext@qdot - jhnow@qdot)/dt
    def getCoM(self):
        model = self.model
        data = self.data
        q = self.q
        qdot = self.qdot
        com = pin.centerOfMass(model,data,q,False)
        self.com = com
        Jcom = pin.jacobianCenterOfMass(model,data,q,False)
        vcom = Jcom@qdot
        return com,vcom
    def tsid(self):
        model = self.model
        data = self.data
        q = self.q
        qdot = self.qdot        
        M = pin.crba(model, data, q)
        c = pin.nle(model, data, q, qdot)
        pin.forwardKinematics(model,data,q,qdot,np.zeros(model.nv))
        pin.framesForwardKinematics(model,data,q) #很奇怪，上边都需要这个，这里经过尝试反而不需要了
        # pin.computeJointJacobians(model,data,q)
        com = pin.centerOfMass(model,data,q,False)
        self.com = com
        Jcom = pin.jacobianCenterOfMass(model,data,q,False)
        vcom = Jcom@qdot

        cN = self.cN
        clist = self.clist
        pN = self.pN
        plist = self.plist
        Pqdd = np.zeros((model.nv+3*cN,model.nv+3*cN))
        qqdd = np.zeros(model.nv+3*cN)
        Wf =  self.Wf
        Wtau = self.Wtau
        contactposition = np.zeros(3*cN)
        for i in range(cN):
            contactposition[3*i:3*(i+1)] = data.oMf[model.getFrameId(clist[i])].translation
        contactvelocity = np.zeros(3*cN)
        for i in range(cN):
            contactvelocity[3*i:3*(i+1)] = pin.getFrameVelocity(model, data, model.getFrameId(clist[i]), pin.LOCAL_WORLD_ALIGNED).linear
        Jdv = np.zeros(3*cN)
        for i in range(cN):
            Jdv[3*i:3*(i+1)] = pin.getFrameAcceleration(model,data,model.getFrameId(clist[i]),pin.LOCAL_WORLD_ALIGNED).linear
        o_Jtool = np.zeros((3*cN,model.nv))
        for i in range(cN):
            o_Jtool[3*i:3*(i+1),:] = pin.computeFrameJacobian(model,data,q,model.getFrameId(clist[i]), pin.LOCAL_WORLD_ALIGNED)[:3,:]
        pcrossm = np.zeros((3,3*cN))
        coml = np.tile(com,cN)
        rforcross = contactposition - coml
        for i in range(cN):
            pcrossm[0,3*i+1] = -rforcross[3*i+2]
            pcrossm[1,3*i+0] = rforcross[3*i+2]
            pcrossm[0,3*i+2] = rforcross[3*i+1]
            pcrossm[2,3*i+0] = -rforcross[3*i+1]
            pcrossm[1,3*i+2] = -rforcross[3*i+0]
            pcrossm[2,3*i+1] = rforcross[3*i+0]
        cftolinf = np.tile(np.identity(3),cN)
        Jfmap = np.block([[cftolinf],[pcrossm]])


        if self.es_flag != 0:
            Jes = np.block([M[6:,:],-o_Jtool.T[6:,:]])
            esbias = -c[6:]
            Pqdd += Jes.T@self.Wtau@Jes
            qqdd += -Jes.T@self.Wtau@esbias
        if self.com_force_task_flag != 0:
            task_a = -160*(com - self.com_task_qobj) - 25*(vcom - self.com_task_vobj) #110,25;190,45
            if self.com_vonly:
                task_a = - 25*(vcom - self.com_task_vobj)
            if task_a[2] < -8.0:
                task_a[2] = -8.0
            # self.clamp(task_a[0],-15,15)
            # self.clamp(task_a[1],-15,15)
            clfb = data.mass[0]*(task_a - np.array([0,0,-9.8]))
            Jclf = np.block([np.zeros((3,self.nv)),cftolinf])
            Pqdd += Jclf.T@self.Wclf@Jclf
            qqdd += -Jclf.T@self.Wclf@clfb
        if self.com_torque_task_flag != 0:
            pin.computeCentroidalMap(model,data,q)
            Jhg = data.Ag[3:,:]
            hg = Jhg@qdot
            # print(hg)
            task_ah = - 30*(hg - self.hg_obj) #25
            crtb = task_ah
            Jcrt = np.block([np.zeros((3,self.nv)),pcrossm])
            Pqdd += Jcrt.T@self.Wcrt@Jcrt
            qqdd += -Jcrt.T@self.Wcrt@crtb
        if self.com_task_flag != 0:
            Jcom = pin.jacobianCenterOfMass(model,data,q,False)
            Jcomf = np.block([Jcom, np.zeros((3,3*cN))])
            Jcomdotqdot = self.getJcomdotqdot(q,qdot)
            com = pin.centerOfMass(model,data,q,False)
            vcom = Jcom@qdot
            task_a = -160*(com - self.com_task_qobj) - 25*(vcom - self.com_task_vobj) #110,25;100,25
            if self.com_vonly:
                task_a = - 25*(vcom - self.com_task_vobj)
            if task_a[2] < -8.0:
                task_a[2] = -8.0
            # self.clamp(task_a[0],-15,15)
            # self.clamp(task_a[1],-15,15)
            Pqdd += Jcomf.T@self.com_task_W@Jcomf
            qqdd += -Jcomf.T@self.com_task_W@(task_a-Jcomdotqdot)
        if self.angular_momenta_flag != 0:
            pin.computeCentroidalMap(model,data,q)
            Jhg = data.Ag[3:,:]
            Jhgf = np.block([Jhg, np.zeros((3,3*cN))])
            Jhdotqdot = self.getJhdotqdot(q,qdot)
            hg = Jhg@qdot
            # print(hg)
            task_ah = - self.kdhg*(hg - self.hg_obj)
            Pqdd += Jhgf.T@self.hgW@Jhgf
            qqdd += -Jhgf.T@self.hgW@(task_ah - Jhdotqdot)
        if self.fr_flag != 0:
            Jfr = np.zeros((3,model.nv+3*cN))
            Jfr[:,3:6] = np.identity(3)
            uvq = q.copy()
            uvq[3:7] = self.fr_obj
            task_afr = 30*pin.difference(model,q,uvq)[3:6] - 15*qdot[3:6] #20,15
            Pqdd += Jfr.T@self.fr_W@Jfr
            qqdd += -Jfr.T@self.fr_W@task_afr
        if self.pN != 0:
            #没有调整J，没有用jpdotqdot
            Jp = np.zeros((3*pN,model.nv))
            for i in range(pN):
                Jp[3*i:3*(i+1),:] = pin.computeFrameJacobian(model,data,q,model.getFrameId(plist[i]), pin.LOCAL_WORLD_ALIGNED)[:3,:]
            # Jp = Jp@self.pSM
            Jpdotqdot = np.zeros(3*pN)
            for i in range(pN):
                Jpdotqdot[3*i:3*(i+1)] = pin.getFrameAcceleration(model,data,model.getFrameId(plist[i]),pin.LOCAL_WORLD_ALIGNED).linear
            pposition = np.zeros(3*pN)
            for i in range(pN):
                pposition[3*i:3*(i+1)] = data.oMf[model.getFrameId(plist[i])].translation
            pvelocity = np.zeros(3*pN)
            for i in range(pN):
                pvelocity[3*i:3*(i+1)] = pin.getFrameVelocity(model, data, model.getFrameId(plist[i]), pin.LOCAL_WORLD_ALIGNED).linear
            task_pa = - 100*(pposition - self.pobj) - 25*(pvelocity - self.pvobj) #20,5
            Pqdd += Jp.T@self.pW@Jp
            qqdd += -(Jp.T)@self.pW@task_pa
        if self.jN != 0:
            Jj = np.zeros((self.jN,self.nv+3*cN))
            for j in range(self.jN):
                Jj[j,self.jlist[j]] = 1
            qhere = np.array([q[i+1] for i in self.jlist])
            qdhere = np.array([qdot[i] for i in self.jlist])
            task_ja = -self.kpj*(qhere - self.qobj) - self.kdj*qdhere
            # task_jv = qdhere + task_ja*self.dt
            Pqdd += Jj.T@self.jW@Jj
            qqdd += -(Jj.T)@self.jW@task_ja
        if self.jzs_flag != 0:
            Jjzs = np.block([np.zeros((self.nv-6,6)),np.identity(self.nv-6), np.zeros((self.nv-6,3*cN))])
            task_jzsq = -10*(qdot[6:])
            Pqdd += Jjzs.T@self.jzs_W@Jjzs
            qqdd += -(Jjzs.T)@self.jzs_W@task_jzsq
        if self.stablize_contact_flag != 0:
            lrcl = len(self.rcl)
            Jsc = np.zeros((lrcl,self.nv))
            for i in range(lrcl):
                Jsc[i,:] = pin.computeFrameJacobian(model,data,q,model.getFrameId(self.rcl[i]), pin.LOCAL_WORLD_ALIGNED)[2,:]
            qcv = Jsc@qdot
            self.Wsc = 40*np.identity(lrcl)
            for i in range(lrcl):
                if qcv[i]<-0.2:
                    self.Wsc[i,i]=0
            # czvsc = contactvelocity[2::3]
            # qsc_a = - 25*(czvsc+0.1)
            qsc_a = -80*(qcv - np.array([-0.2]*lrcl))#czvsc + qsc_a*self.dt
            Pqdd[:self.nv,:self.nv] += Jsc.T@self.Wsc@Jsc
            qqdd[:self.nv] += -(Jsc.T)@self.Wsc@qsc_a
        if self.trl[0] == 0:
            idrf = model.getFrameId('link3')
            oMgoal = pin.SE3(pin.Quaternion(1, 0, 0, 0).normalized().matrix(), np.array([com[0]-vcom[0]*0.0, -0.15, com[2]-0.79+vcom[2]*0.05]))
            oMtool = data.oMf[idrf]
            tool_nu = pin.log(oMtool.inverse()*oMgoal).vector
            tool_Jtool = pin.computeFrameJacobian(model, data, q, idrf)
            tool_v = tool_Jtool@qdot
            task_rca = (tool_nu/0.05 - tool_v)*50#(tool_nu/self.dt - tool_v)*3#/self.dt #150*(tool_nu) - 20*tool_v
            rcW = 1*np.identity(6) #0.1
            # rcW[2,2] = 0
            Pqdd[:self.nv,:self.nv] += tool_Jtool.T@rcW@tool_Jtool
            qqdd[:self.nv]+= -(tool_Jtool.T)@rcW@(task_rca)
        if self.trl[1] == 0:
            idrf = model.getFrameId('link6')
            oMgoal = pin.SE3(pin.Quaternion(1, 0, 0, 0).normalized().matrix(), np.array([com[0]-vcom[0]*0.0, 0.15, com[2]-0.79+vcom[2]*0.05]))
            oMtool = data.oMf[idrf]
            tool_nu = pin.log(oMtool.inverse()*oMgoal).vector
            tool_Jtool = pin.computeFrameJacobian(model, data, q, idrf)
            tool_v = tool_Jtool@qdot
            task_lca = (tool_nu/0.05 - tool_v)*50#(tool_nu/self.dt - tool_v)*3#/self.dt #150*(tool_nu) - 20*tool_v
            lcW = 1*np.identity(6) #0.1
            # lcW[2,2] = 0
            Pqdd[:self.nv,:self.nv] += tool_Jtool.T@lcW@tool_Jtool
            qqdd[:self.nv] += -(tool_Jtool.T)@lcW@(task_lca)

        P  = Pqdd + 0.001*np.identity(self.nv+3*cN)#TA.T@W@TA # sforqddreg.T@self.Wreg@sforqddreg +
        oq = qqdd#TA.T@W@sb
        A = np.block([[M[:6,:],-o_Jtool.T[:6,:]],[o_Jtool,0.001*np.identity(3*cN)]]) #np.zeros((3*cN,3*cN))
        A[:6,:6] += 0.00001*np.identity(6)
        b = np.concatenate((-c[:6],-Jdv- 5*(contactvelocity)))#- 20*(contactposition - origincontactposition) - 5*(contactvelocity)
        maxset = 100000
        lb = np.array([-maxset]*6+[-15000]*14+[-19000,-19000,0.1]*(cN)) #10.1
        ub = np.array([ maxset]*6+[ 15000]*14+[ 19000]*(3*cN))
        self.x = proxqp_solve_qp(P,oq,A=A,b=b,lb=lb,ub=ub,eps_abs=1e-7)
        self.cf = self.x[self.nv:]
        self.tau = M[6:,:]@self.x[:self.nv] + (c[6:]) - o_Jtool.T[6:,:]@self.cf
        for i in range(cN):
            if self.clist  and self.x[self.nv+2+3*i] <= -0.00001:
                print('negtive force:',self.x[self.nv+2+3*i])