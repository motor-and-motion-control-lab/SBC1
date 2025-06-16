import numpy as np
from math import cos, sin
from numpy.linalg import pinv
from scipy.linalg import block_diag
from qpsolvers import solve_qp
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
from numpy.linalg import pinv
import mujoco

class TaskListc:
    def __init__(self,model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.nv = model.nv
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

        self.fixed_body_num = 0
        self.allconsidered_fixed_body_list = ('link3','link6')
        self.fixed_body_list = []
        self.fixed_body_id_list = []
        self.vertex_point_site_num = 0
        self.vertex_point_site_list = []
        
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
        self.fr_obj = np.array([1,0,0,0]) #pin.Quaternion(0.7, 0.2, 0.2, 0.6).normalized()#
        self.fr_W = np.diag([0.5]*3) #50

        self.x = np.zeros(model.nv+3*self.cN+model.nv-6)
        self.tau = np.zeros(model.nv-6)
        self.cf = np.zeros(3*8)
        
        self.Wreg = 0.001*np.identity(model.nv)
        for i in range(6):
            self.Wreg[i,i] = 0.0001
        self.bend_knee_flag = 0
        self.es_flag = 1
        self.Wtau = np.diag([0.018]*(model.nv-6))
        self.jzs_flag = 0
        self.jzs_W = 0.01*np.identity(self.nv-6)

        self.fs = 0
        self.concou = np.zeros(len(self.fclist))

        self.trl = [1,1]
        self.rc_flag = 0
        self.lc_flag = 0
        self.right_foot_pos_target = None
        self.left_foot_pos_target = None

        self.com = np.zeros(3)
        self.miu = 0.7
        
    # def setCPNJ(self,cN):
    #     self.cN = cN
    #     self.Jc = np.zeros((3*cN,self.nv))
    def setCPNL(self,clist):
        self.cN = len(clist)
        self.clist = clist
        self.invR = np.diag([200]*3*self.cN) #1000就不行了，注意这里是实际的invR多乘以dt后的，所以很大了
        self.Wf = np.diag([10,10,10]*(self.cN))
        # self.concou = np.zeros(self.cN)
    def addComTask(self,W=[20]*3):
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
        self.data.qpos[:] = q
        self.data.qvel[:] = qdot

    def get_body_list_jac(self, bodyidlist):
        body_list_oJ = np.zeros((6*len(bodyidlist),self.model.nv))
        for i in range(len(bodyidlist)):
            jacp = np.zeros((3,self.model.nv))
            jacr = np.zeros((3,self.model.nv))
            mujoco.mj_jacBody(self.model,self.data,jacp,jacr,bodyidlist[i])
            body_list_oJ[6*i:6*i+3,:] = jacp
            body_list_oJ[6*i+3:6*i+6,:] = jacr
        return body_list_oJ
    
    def getJcomdotqdot(self, dt=1e-6):
        mujoco.mj_fwdPosition(self.model, self.data)
        com0 = self.data.subtree_com[0].copy()

        J_com = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, J_com, 0)

        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()

        qpos_next = qpos0.copy()
        mujoco.mj_integratePos(self.model, qpos_next, qvel0, dt)
        self.data.qpos[:] = qpos_next
        mujoco.mj_fwdPosition(self.model, self.data)
        
        J_com_next = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, J_com_next, 0)
        
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = qvel0
        mujoco.mj_fwdPosition(self.model, self.data)
        
        J_dot = (J_com_next - J_com) / dt
        
        jdotqdot = J_dot @ self.data.qvel
    
        return jdotqdot
    
    def getJhdotqdot(self, dt=1e-6):
        mujoco.mj_fwdPosition(self.model, self.data)
        J_am = np.zeros((3, self.model.nv))
        mujoco.mj_angmomMat(self.model, self.data, J_am, 0)

        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()

        qpos_next = qpos0.copy()
        mujoco.mj_integratePos(self.model, qpos_next, qvel0, dt)
        self.data.qpos[:] = qpos_next
        mujoco.mj_fwdPosition(self.model, self.data)
        
        J_am_next = np.zeros((3, self.model.nv))
        mujoco.mj_angmomMat(self.model, self.data, J_am_next, 0)
        
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = qvel0
        mujoco.mj_fwdPosition(self.model, self.data)
        
        J_dot = (J_am_next - J_am) / dt
        
        jdotqdot = J_dot @ self.data.qvel
    
        return jdotqdot
    
    def getJfixeddotqdot(self, dt=1e-6):
        mujoco.mj_fwdPosition(self.model, self.data)
        J_fixed = self.get_body_list_jac(self.fixed_body_id_list)

        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()

        qpos_next = qpos0.copy()
        mujoco.mj_integratePos(self.model, qpos_next, qvel0, dt)
        self.data.qpos[:] = qpos_next
        mujoco.mj_fwdPosition(self.model, self.data)
        
        J_fixed_next = self.get_body_list_jac(self.fixed_body_id_list)
        
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = qvel0
        mujoco.mj_fwdPosition(self.model, self.data)
        
        J_dot = (J_fixed_next - J_fixed) / dt
        
        fixed_body_Jdv = J_dot @ self.data.qvel
    
        return fixed_body_Jdv
    
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
        # 角动量
        self.angular_momenta_flag = 0
        self.hgW = np.diag([10]*3)
        self.kdhg = 65
        self.hg_obj = np.zeros(3) 
        # 上身直立
        self.fr_flag = 1
        self.fr_W = np.diag([5]*3)
        # 节能
        self.es_flag = 1
        self.Wtau = np.diag([0.018]*(self.nv-6))
        # 不动
        self.jzs_flag = 0
        self.jzs_W = 0.01*np.identity(self.nv-6)

    def ret(self):
        self.es_flag = 0
        self.com_task_flag = 0
        self.angular_momenta_flag = 0
        self.fr_flag = 0
        self.pN = 0
        self.jN = 0
        self.jzs_flag = 0

    def get_joint_qpos_indices(self, joint_name):
        # 获取关节 ID
        model = self.model
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found in model")
        
        # 获取关节的 qpos 起始地址
        qpos_adr = model.jnt_qposadr[joint_id]
        
        # 根据关节类型确定索引范围
        joint_type = model.jnt_type[joint_id]
        
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return [qpos_adr]  # 单自由度关节
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return list(range(qpos_adr, qpos_adr + 4))  # 球关节 (四元数)
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return list(range(qpos_adr, qpos_adr + 7))  # 自由关节 (3位置 + 4四元数)
        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

    def tsid(self):
        model = self.model
        data = self.data
        self.setStates(self.q,self.qdot)
        self.M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        self.c = self.data.qfrc_bias

        mujoco.mj_forward(model,data)
        self.com = data.subtree_com[1]
        J_com = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, J_com, 0)
        vcom = J_com@self.qdot
        J_am = np.zeros((3, self.model.nv))
        mujoco.mj_angmomMat(self.model, self.data, J_am, 0)
        # vh = J_am @ self.data.qvel
        
        self.fixed_body_id_list = [model.body(name).id for name in self.fixed_body_list]
        self.fixed_body_num = len(self.fixed_body_list)
        fixed_body_pos = np.zeros(3*self.fixed_body_num)
        for i in range(self.fixed_body_num):
            bodyid = self.fixed_body_id_list[i]
            fixed_body_pos[3*i:3*(i+1)] = data.xpos[bodyid] #for cpp: data.xpos[3*bodyid:3*(bodyid+1)]
        fixed_body_quat = np.zeros(4*self.fixed_body_num)
        for i in range(self.fixed_body_num):
            bodyid = self.fixed_body_id_list[i]
            fixed_body_quat[4*i:4*(i+1)] = data.xquat[bodyid] #4*bodyid:4*(bodyid+1)
        # fixed_body_oJ = np.zeros((6*self.fixed_body_num, model.nv))
        fixed_body_velocity = np.zeros(6*self.fixed_body_num)
        # for i in range(self.fixed_body_num):
        #     bodyid = self.fixed_body_id_list[i]
        #     jacp = np.zeros((3,model.nv))
        #     jacr = np.zeros((3,model.nv))
        #     mujoco.mj_jacBody(model,data,jacp,jacr,bodyid)
        #     fixed_body_oJ[6*i:6*i+3,:] = jacp
        #     fixed_body_oJ[6*i+3:6*i+6,:] = jacr
        #     fixed_body_velocity[6*i:6*(i+1)] = fixed_body_oJ@self.qdot
        fixed_body_oJ = self.get_body_list_jac(self.fixed_body_id_list)
        fixed_body_velocity = fixed_body_oJ@self.qdot
        fixed_body_Jdv = self.getJfixeddotqdot()
        self.vertex_point_site_num = len(self.vertex_point_site_list)
        vertex_point_site_jac = np.zeros((3*self.vertex_point_site_num, model.nv))
        for i in range(self.vertex_point_site_num):
            mujoco.mj_jacSite(self.model,self.data,vertex_point_site_jac[3*i:3*(i+1),:],None,model.site(self.vertex_point_site_list[i]).id)
        pN = self.pN
        plist = self.plist
        Pqdd = np.zeros((model.nv+3*self.vertex_point_site_num,model.nv+3*self.vertex_point_site_num))
        qqdd = np.zeros(model.nv+3*self.vertex_point_site_num)

        if self.es_flag != 0:
            Jes = np.block([self.M[6:,:],-vertex_point_site_jac.T[6:,:]])
            # print(Jes)
            esbias = -self.c[6:]
            Pqdd += Jes.T@self.Wtau@Jes
            qqdd += -Jes.T@self.Wtau@esbias
        if self.com_task_flag != 0:
            Jcom = J_com
            Jcomf = np.block([Jcom, np.zeros((3,3*self.vertex_point_site_num))])
            Jcomdotqdot = self.getJcomdotqdot()
            com = self.com
            vcom = Jcom@self.qdot
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
            Jhg = J_am
            Jhg = data.Ag[3:,:]
            Jhgf = np.block([Jhg, np.zeros((3,3*self.vertex_point_site_num))])
            Jhdotqdot = self.getJhdotqdot()
            hg = Jhg@self.qdot
            # print(hg)
            task_ah = - self.kdhg*(hg - self.hg_obj)
            Pqdd += Jhgf.T@self.hgW@Jhgf
            qqdd += -Jhgf.T@self.hgW@(task_ah - Jhdotqdot)
        if self.fr_flag != 0:
            Jfr = np.zeros((3,model.nv+3*self.vertex_point_site_num))
            Jfr[:,3:6] = np.identity(3)
            uvq = self.q.copy()
            uvq[3:7] = self.fr_obj
            posdifference = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model,posdifference,1,self.q,uvq)
            task_afr = 30*posdifference[3:6] - 15*self.qdot[3:6] #20,15
            Pqdd += Jfr.T@self.fr_W@Jfr
            qqdd += -Jfr.T@self.fr_W@task_afr
        if self.pN != 0:
            #没有调整J，没有用jpdotqdot
            Jp = np.zeros((3*pN,model.nv))
            # for i in range(pN):
            #     Jp[3*i:3*(i+1),:] = pin.computeFrameJacobian(model,data,q,model.getFrameId(plist[i]), pin.LOCAL_WORLD_ALIGNED)[:3,:]
            # # Jp = Jp@self.pSM
            # Jpdotqdot = np.zeros(3*pN)
            # for i in range(pN):
            #     Jpdotqdot[3*i:3*(i+1)] = pin.getFrameAcceleration(model,data,model.getFrameId(plist[i]),pin.LOCAL_WORLD_ALIGNED).linear
            # pposition = np.zeros(3*pN)
            # for i in range(pN):
            #     pposition[3*i:3*(i+1)] = data.oMf[model.getFrameId(plist[i])].translation
            # pvelocity = np.zeros(3*pN)
            # for i in range(pN):
            #     pvelocity[3*i:3*(i+1)] = pin.getFrameVelocity(model, data, model.getFrameId(plist[i]), pin.LOCAL_WORLD_ALIGNED).linear
            # task_pa = - 100*(pposition - self.pobj) - 25*(pvelocity - self.pvobj) #20,5
            # Pqdd += Jp.T@self.pW@Jp
            # qqdd += -(Jp.T)@self.pW@task_pa
        if self.jN != 0:
            Jj = np.zeros((self.jN,self.nv+3*self.vertex_point_site_num))
            for j in range(self.jN):
                Jj[j,self.jlist[j]] = 1
            qhere = np.array([self.q[i+1] for i in self.jlist])
            qdhere = np.array([self.qdot[i] for i in self.jlist])
            task_ja = -self.kpj*(qhere - self.qobj) - self.kdj*qdhere
            Pqdd += Jj.T@self.jW@Jj
            qqdd += -(Jj.T)@self.jW@task_ja
        if self.jzs_flag != 0:
            Jjzs = np.block([np.zeros((self.nv-6,6)),np.identity(self.nv-6), np.zeros((self.nv-6,3*self.vertex_point_site_num))])
            task_jzsq = -10*(self.qdot[6:])
            Pqdd += Jjzs.T@self.jzs_W@Jjzs
            qqdd += -(Jjzs.T)@self.jzs_W@task_jzsq

        if self.trl[0] == 0:
            idrf = model.body('link3').id
            tool_nu = np.zeros(6)
            tool_nu[:3] = self.right_foot_pos_target - data.body(idrf).xpos
            rf_neg_quat = np.zeros(4)
            rf_error_quat = np.zeros(4)
            mujoco.mju_negQuat(rf_neg_quat, data.body(idrf).xquat)
            mujoco.mju_mulQuat(rf_error_quat, np.array([1,0,0,0]), rf_neg_quat)
            mujoco.mju_quat2Vel(tool_nu[3:], rf_error_quat, 1.0)
            Jrf = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, Jrf[:3], Jrf[3:], idrf)
            tool_v = Jrf@self.qdot
            task_rca = (tool_nu/0.05 - tool_v)*50#(tool_nu/self.dt - tool_v)*3#/self.dt #150*(tool_nu) - 20*tool_v
            rcW = 1*np.identity(6) #0.1
            # rcW[2,2] = 0
            Pqdd[:self.nv,:self.nv] += Jrf.T@rcW@Jrf
            qqdd[:self.nv]+= -(Jrf.T)@rcW@(task_rca)
        if self.trl[1] == 0:
            idlf = model.body('link6').id
            tool_nu = np.zeros(6)
            tool_nu[:3] = self.left_foot_pos_target - data.body(idlf).xpos
            lf_neg_quat = np.zeros(4)
            lf_error_quat = np.zeros(4)
            mujoco.mju_negQuat(lf_neg_quat, data.body(idlf).xquat)
            mujoco.mju_mulQuat(lf_error_quat, np.array([1,0,0,0]), lf_neg_quat)
            mujoco.mju_quat2Vel(tool_nu[3:], lf_error_quat, 1.0)
            Jlf = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, Jlf[:3], Jlf[3:], idlf)
            tool_v = Jlf@self.qdot
            task_lca = (tool_nu/0.05 - tool_v)*50#(tool_nu/self.dt - tool_v)*3#/self.dt #150*(tool_nu) - 20*tool_v
            lcW = 1*np.identity(6) #0.1
            # lcW[2,2] = 0
            Pqdd[:self.nv,:self.nv] += Jlf.T@lcW@Jlf
            qqdd[:self.nv] += -(Jlf.T)@lcW@(task_lca)

        P  = Pqdd + 0.001*np.identity(self.nv+3*self.vertex_point_site_num)
        oq = qqdd
        A = np.block([[self.M[:6,:],-vertex_point_site_jac.T[:6,:]],[fixed_body_oJ,np.zeros((6*self.fixed_body_num,3*self.vertex_point_site_num))]]) #0.00*np.identity(3*self.vertex_point_site_num)
        # A[:6,:6] += 0.00001*np.identity(6)
        b = np.concatenate((-self.c[:6],-fixed_body_Jdv- 5*(fixed_body_velocity)))#- 20*(contactposition - origincontactposition) - 5*(contactvelocity)
        maxset = 1000
        lb = np.array([-maxset]*6+[-1500]*14+[-3900,-3900,0.0]*(self.vertex_point_site_num))
        ub = np.array([ maxset]*6+[ 1500]*14+[ 3900]*(3*self.vertex_point_site_num))
        indice_kneer = self.get_joint_qpos_indices('KNEERY')[0]
        indice_kneel = self.get_joint_qpos_indices('KNEELY')[0]
        if self.q[indice_kneer] <=0:
            lb[indice_kneer-1] = 2*(-self.q[indice_kneer]-self.qdot[indice_kneer-1]*self.dt)/self.dt/self.dt
        if self.q[indice_kneel] <=0:
            lb[indice_kneel-1] = 2*(-self.q[indice_kneel]-self.qdot[indice_kneel-1]*self.dt)/self.dt/self.dt
        G_elem = np.array([[1,0,-self.miu],[0,1,-self.miu],[-1,0,-self.miu],[0,-1,-self.miu]])
        G_elem2 = np.kron(np.eye(self.vertex_point_site_num), G_elem)
        G_ineq = np.block([np.zeros((4*self.vertex_point_site_num,model.nv)), G_elem2])
        h_ineq = np.zeros(4*self.vertex_point_site_num)
        self.x = proxqp_solve_qp(P,oq,G=G_ineq,h=h_ineq,A=A,b=b,lb=lb,ub=ub,eps_abs=1e-7)
        self.cf = self.x[self.nv:]
        self.tau = self.M[6:,:]@self.x[:self.nv] + (self.c[6:]) - vertex_point_site_jac.T[6:,:]@self.cf

if __name__ == '__main__':
    modelfilename = 'test4.xml'
    m = mujoco.MjModel.from_xml_path(modelfilename)
    d = mujoco.MjData(m)
    model = m
    data = d
    mujoco.mj_forward(m,d)
    t1 = TaskListc(model)
    t1.setStates(data.qpos,data.qvel)
    print(t1.q)
    print(t1.qdot)
    print(t1.data.qpos)
    print(t1.data.qvel)
    print('*******************contact*********************')
    t1.fixed_body_list = ['link3','link6']
    t1.vertex_point_site_list = ['t{}'.format(i) for i in range(1, 9)]
    print(t1.vertex_point_site_list)
    print(t1.es_flag)
    print(t1.fixed_body_list)
    print('*******************task*********************')
    t1.dt = 0.005
    t1.sm1(0,0)
    t1.tsid()
    print('tau:',t1.x)