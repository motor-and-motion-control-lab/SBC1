from calendar import c
import numpy as np
import matplotlib.pylab as plt
from webcolors import CSS21; plt.ion()

def quintic_trajectory(start, end, duration, t):
    """生成五次多项式轨迹（位置、速度、加速度在起点终点都为零）"""
    t_n = t / duration  # 归一化时间 [0,1]
    # 五次多项式混合系数
    blend = 10*t_n**3 - 15*t_n**4 + 6*t_n**5
    return start + (end - start)*blend

def generate_swing_foot_trajectory(start_pos, end_pos, max_height, duration, dt=0.01):
    """生成协调的摆动脚轨迹（所有方向使用五次多项式）"""
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    
    # print('dur:',duration)
    # print('int:',int(duration / dt))
    duration += 0.0000000001
    num_points = int(duration / dt) + 1
    # print('n:',num_points)
    time_points = np.linspace(0, duration, num_points)
    
    # 计算中间最高点（在起点和终点的最高值基础上增加max_height）
    peak_height = max(start_pos[2], end_pos[2]) + max_height
    mid_time = duration / 2
    
    # 生成各轴轨迹
    x_traj = quintic_trajectory(start_pos[0], end_pos[0], duration, time_points)
    y_traj = quintic_trajectory(start_pos[1], end_pos[1], duration, time_points)
    
    # Z轴分为两段：上升段和下降段
    z_traj = np.zeros_like(time_points)
    rise_mask = time_points <= mid_time
    fall_mask = ~rise_mask
    
    # 上升段（从start_z到peak_height）
    z_traj[rise_mask] = quintic_trajectory(
        start_pos[2], peak_height, mid_time, time_points[rise_mask])
    
    # 下降段（从peak_height到end_z）
    z_traj[fall_mask] = quintic_trajectory(
        peak_height, end_pos[2], duration-mid_time, time_points[fall_mask]-mid_time)
    
    trajectory = np.column_stack((x_traj, y_traj, z_traj))
    return time_points, trajectory

class ContactSchedule:
    def __init__(self) -> None:
        self.sch = []
        self.finaltime = 0
        self.hl = 0.13
        self.hw = 0.04
        self.ff = np.array([self.hl,0,0])
        self.fb = np.array([-self.hl,0,0])
        self.N = 200
        self.dt = 0.01
        self.xhist = []
        self.uhist = []
        self.cx = np.zeros((2,self.N))
        self.cy = np.zeros((2,self.N))
        self.px = np.zeros(self.N-1)
        self.py = np.zeros(self.N-1)
        self.pxobjs = 0.0*np.ones(self.N-1)
        self.pyobjs = 0.0*np.ones(self.N-1)

        self.swing_sch = []
        self.last_rf_pos = None
        self.last_lf_pos = None
        self.rf_pos_list = np.zeros((3,self.N))
        self.lf_pos_list = np.zeros((3,self.N))
        self.deadband_time = 0.04
    def pushContactList(self,duration,fixed_body_list,vertex_point_site_list,cenpos):
        self.sch.append([self.finaltime,fixed_body_list,vertex_point_site_list,cenpos])
        self.finaltime += duration
    # def insertContactList(self,time,clist,clp):
    #     for i in range(len(self.sch)):
    #         if time > self.sch[i][0]:
    #             self.sch.insert(i+1,[time,clist,clp])
    def getCurrentsid(self,time):
        for i in range(len(self.sch)):
            if time < self.sch[i][0]:
                return i-1
        return len(self.sch)-1
    # def getCurrentswingsid(self,time):
    #     for i in range(len(self.swing_sch)):
    #         if time < self.swing_sch[i][0]:
    #             return i-1
    #     return len(self.swing_sch)-1
    # def getTimeBefore(self,time):
    #     time_before = 0
    #     for i in range(len(self.swing_sch)):
    #         if time < self.swing_sch[i][0]:
    #             return i-1
    #         time_before += self.swing_sch[i][0]
    #     return len(self.swing_sch)-1
                
    def printInfo(self):
        # print('time','\t','clist','\t','clp')
        cll1 = []
        for i in range(len(self.sch)):
            cll1.append(len(str(self.sch[i][1])))
        cll2 = []
        for i in range(len(self.sch)):
            cll2.append(len(str(self.sch[i][2])))
        # print(max(cll))
        for i in range(len(self.sch)):
            # print(self.sch[i][0],'\t',self.sch[i][1],'\t',self.sch[i][2])
            print('%.2f'%self.sch[i][0],'\t','%-*s'%(max(cll1)+4,self.sch[i][1]),'\t','%-*s'%(max(cll2)+4,self.sch[i][2]), self.sch[i][3])
    def printSwingSch(self):
        for i in range(len(self.swing_sch)):
            print(i,self.swing_sch[i])
    def pushDS2SSOnR(self,rf,lf):
        ff = self.ff
        fb = self.fb
        self.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(rf+lf)/2)
        self.pushContactList(0.3,['t1','t2','t3','t4','t5','t7'],(2*rf+lf+ff)/3)
        self.pushContactList(0.3,['t1','t2','t3','t4'],(rf))
    def pushDS2SSOnL(self,rf,lf):
        ff = self.ff
        fb = self.fb
        self.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(rf+lf)/2)
        self.pushContactList(0.3,['t1','t3','t5','t6','t7','t8'],(rf+2*lf+ff)/3)
        self.pushContactList(0.3,['t5','t6','t7','t8'],(lf))
    def pushDS(self,rf,lf,time=0.3):
        self.pushContactList(time,['link3','link6'],['t1','t2','t3','t4','t5','t6','t7','t8'],(rf+lf)/2)
        self.swing_sch.append([time,None,None,None,rf,lf])
        self.last_rf_pos = rf
        self.last_lf_pos = lf
        # begin_index = (self.finaltime/self.dt)
        # end_index = (self.finaltime+time)/self.dt
        # if end_index 
        # self.rf_pos_list[:,:]
    def pushSSOnR(self,rf,lfdes,time=0.3):
        self.pushContactList(time,['link3'],['t1','t2','t3','t4'],rf)
        self.swing_sch.append([time,['link6'],self.last_lf_pos,lfdes,rf,lfdes])
        self.last_rf_pos = rf
        self.last_lf_pos = lfdes
    def pushSSOnL(self,rfdes,lf,time=0.3):
        self.pushContactList(time,['link6'],['t5','t6','t7','t8'],lf)
        self.swing_sch.append([time,['link3'],self.last_rf_pos,rfdes,rfdes,lf])
        self.last_rf_pos = rfdes
        self.last_lf_pos = lf
    # def set_sch(self,duration,rfsch,lfsch):
        
    
    def LIPLQR(self,uobjs,final_xobj,x0):
        N = self.N
        dt = self.dt
        P = np.zeros((2,2,N))
        p = np.zeros((2,N))
        K = np.zeros((1,2,N-1))
        d = np.zeros(N-1)
        h = np.ones(N-1)

        Q = np.diag([0.0,0.0])
        Qn = np.diag([10.0,0.0])
        # R = np.array([0.01])# 0.1结尾就不会上翘，N=200就能看出收敛
        R = 1 #0.01
        g = 9.8
        P[:,:,N-1] = Qn
        p[:,N-1] = Qn@(-final_xobj)

        #Backward Riccati recursion
        for k in range(N-2, -1, -1):
            discrete_omega_square = g*dt/h[k]
            A = np.array([[1,dt],[discrete_omega_square,1]])
            B = np.array([[0],[-discrete_omega_square]])
            gx = Q@(-final_xobj)+ A.T@p[:,k+1]
            gu = R*(-uobjs[k])  + B.T@p[:,k+1]
            Gxx = Q + A.T@P[:,:,k+1]@A
            Gxu = A.T@P[:,:,k+1]@B
            Guu = R + B.T@P[:,:,k+1]@B
            # invGuu = np.linalg.inv(Guu)
            K[:,:,k] = (Gxu/Guu).T#invGuu@Gxu #实际是Gux
            # print(gu,Guu)
            d[k] = (gu[0]/Guu[0][0])#.T#invGuu@gu
            P[:,:,k] = Gxx + K[:,:,k].T@Guu@K[:,:,k] - Gxu@K[:,:,k] - K[:,:,k].T@(Gxu.T)
            p[:,k] = gx - (K[:,:,k].T*gu + K[:,:,k].T@Guu*d[k] - Gxu*d[k]).reshape(2)

        #Forward rollout starting at x0
        xhist = np.zeros((2,N))
        xhist[:,0] = x0
        uhist = np.zeros(N-1)
        for k in range(N-1):
            discrete_omega_square = g*dt/h[k]
            A = np.array([[1,dt],[discrete_omega_square,1]])
            B = np.array([[0],[-discrete_omega_square]])
            uhist[k] = (-K[:,:,k]@xhist[:,k] - d[k])[0]
            xhist[:,k+1] = A@xhist[:,k] + (B*uhist[k]).reshape(2)
        return xhist,uhist
    def getxyft(self,time):
        sid = self.getCurrentsid(time)
        x = self.sch[sid][3][0]
        y = self.sch[sid][3][1]
        return x,y
    def getTraj(self,time,cx0,cy0,final_cxobj,final_cyobj):
        N = self.N
        dt = self.dt
        self.pxobjs = 0.0*np.ones(self.N-1)
        self.pyobjs = 0.0*np.ones(self.N-1)
        self.cx = np.zeros((2,self.N))
        self.cy = np.zeros((2,self.N))
        self.px = np.zeros(self.N-1)
        self.py = np.zeros(self.N-1)
        # self.pxobjs = 0.0*np.ones(N-1)
        # final_cxobj = np.array([0.06,0])
        # final_cxobj = final_c[0]
        # self.pyobjs = 0.0*np.ones(N-1)
        # final_cyobj = np.array([0.15,0])
        # final_cyobj = final_c[1]
        for i in range(N-1):
            self.pxobjs[i],self.pyobjs[i] = self.getxyft(time+i*dt)
        self.cx,self.px = self.LIPLQR(self.pxobjs,final_cxobj,cx0)
        self.cy,self.py = self.LIPLQR(self.pyobjs,final_cyobj,cy0)
    def plotTraj(self):
        plt.subplot(3,1,1)
        plt.plot(self.cx[0,:],'gray',label='position of CoM')
        # plt.show()
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(self.px,'blue',label='point of force') #point of force
        plt.plot(self.pxobjs,'yellow',label='objects for point of force') #objects for point of force
        plt.legend(prop = {'size':6})
        # plt.subplot(3,1,3)
        # plt.plot(h,'red',label='height')
        # plt.legend()
        # plt.savefig('images/mpp2tfcls1/comx_trajectories1_mpp2tfcls1.png')

        plt.figure(2)
        plt.subplot(3,1,1)
        plt.plot(self.cy[0,:],'gray',label='position of CoM')
        # plt.show()
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(self.py,'blue',label='point of force') #point of force
        plt.plot(self.pyobjs,'yellow',label='objects for point of force') #objects for point of force
        plt.legend(prop = {'size':6})
        # plt.savefig('images/mpp2tfcls1/comy_trajectories1_mpp2tfcls1.png')

        plt.figure(3)
        plt.subplot(2,1,1)
        plt.plot(self.cx[0,:],self.cy[0,:],'gray',label='position of CoM')
        plt.plot(self.px,self.py,'blue',label='point of force') #point of force
        plt.plot(self.pxobjs,self.pyobjs,'yellow',label='objects for point of force') #objects for point of force
        plt.legend(prop = {'size':6})
        # plt.show()
        # plt.legend()
        # plt.subplot(2,1,2)
        # plt.savefig('images/mpp2tfcls1/com_trajectories1_mpp2tfcls1.png')
        plt.show()
    def getSwingTraj(self):
        allconsidered_fixed_body_list = ('link3','link6')
        # for time in range(self.N*self.dt):
        #     sid = self.getCurrentswingsid(time)
        #     for bodyi in allconsidered_fixed_body_list:
        #         if bodyi in self.swing_sch[sid][1]:
        time = 0
        ind = 0
        for ssi in self.swing_sch:
            sid = self.getCurrentsid(time)
            if ssi[1] and 'link3' in ssi[1]:
                _ , traj = generate_swing_foot_trajectory(ssi[2],ssi[3],0.08,ssi[0]-2*self.deadband_time,self.dt)
                ssi.append(traj)
            else:
                ssi.append(ssi[4])
            if ssi[1] and 'link6' in ssi[1]:
                _ , traj = generate_swing_foot_trajectory(ssi[2],ssi[3],0.08,ssi[0]-2*self.deadband_time,self.dt)
                ssi.append(traj)
            else:
                ssi.append(ssi[5])
            time += ssi[0]
            ind += 1
    def pushDST(self,rf,lf,fpoint,time):
        self.pushContactList(time,['link3','link6'],['t1','t2','t3','t4','t5','t6','t7','t8'],fpoint)
        self.swing_sch.append([time,None,None,None,rf,lf])
        self.last_rf_pos = rf
        self.last_lf_pos = lf
    def get_com_traj_point(self,time,ctrl_dt=None):
        ind = int((time)/self.dt)#int((time-1)/self.dt)
        if ind > self.N-1:
            ind = self.N-1
        com_task_qobj = np.array([self.cx[0,ind],self.cy[0,ind],1])
        com_task_vobj = np.array([self.cx[1,ind],self.cy[1,ind],0])
        if ind > self.N-2:
            com_task_aobj = np.zeros(3)
        else:
            com_task_aobj = (np.array([self.cx[1,ind+1],self.cy[1,ind+1],0]) - com_task_vobj)/self.dt
        return com_task_aobj, com_task_vobj, com_task_qobj
    def get_swing_traj_point(self,time,ctrl_dt=None):
        sid = self.getCurrentsid(time)
        time_begin = self.sch[sid][0]+self.deadband_time
        time_end = self.sch[sid][0] + self.swing_sch[sid][0] -self.deadband_time
        ssi = self.swing_sch[sid]
        lf_state = None
        rf_state = None
        lf_swing_traj_point = []
        rf_swing_traj_point = []
        if ssi[1] and 'link3' in ssi[1] and (time >= time_begin+self.dt/2 and time < time_end-self.dt/2):
            if time - time_begin < self.dt:
                rf_swing_traj_qp = ssi[6][int((time - time_begin)/self.dt)]
            if time_end - time < self.dt:
                rf_swing_traj_qn = ssi[6][int((time - time_begin)/self.dt)]
            rf_swing_traj_qobj = ssi[6][int((time - time_begin)/self.dt)]
            rf_swing_traj_qn = ssi[6][int((time - time_begin)/self.dt)+1]
            rf_swing_traj_qp = ssi[6][int((time - time_begin)/self.dt)-1]
            rf_swing_traj_vobj = (rf_swing_traj_qn - rf_swing_traj_qobj)/2
            rf_swing_traj_vp = (rf_swing_traj_qobj - rf_swing_traj_qp)/2
            rf_swing_traj_aobj = (rf_swing_traj_vobj - rf_swing_traj_vp)/2
            # rf_swing_traj_point.append('swing')
            rf_state = 'swing'
            rf_swing_traj_point.append(rf_swing_traj_qobj)
            rf_swing_traj_point.append(rf_swing_traj_vobj)
            rf_swing_traj_point.append(rf_swing_traj_aobj)
        elif ssi[1] and 'link3' in ssi[1] and time < time_begin+self.dt/2:
            rf_swing_traj_qobj = ssi[6][0]
            rf_swing_traj_vobj = np.zeros(3)
            rf_swing_traj_aobj = np.zeros(3)
            rf_state = 'fixed'
            rf_swing_traj_point.append(rf_swing_traj_qobj)
            rf_swing_traj_point.append(rf_swing_traj_vobj)
            rf_swing_traj_point.append(rf_swing_traj_aobj)
        elif ssi[1] and 'link3' in ssi[1] and time >= time_end-self.dt/2:
            rf_swing_traj_qobj = ssi[6][-1]
            rf_swing_traj_vobj = np.zeros(3)
            rf_swing_traj_aobj = np.zeros(3)
            rf_state = 'fixed'
            rf_swing_traj_point.append(rf_swing_traj_qobj)
            rf_swing_traj_point.append(rf_swing_traj_vobj)
            rf_swing_traj_point.append(rf_swing_traj_aobj)
        else:
            rf_swing_traj_qobj = ssi[6]
            rf_swing_traj_vobj = np.zeros(3)
            rf_swing_traj_aobj = np.zeros(3)
            # rf_swing_traj_point.append('fixed')
            rf_state = 'fixed'
            rf_swing_traj_point.append(rf_swing_traj_qobj)
            rf_swing_traj_point.append(rf_swing_traj_vobj)
            rf_swing_traj_point.append(rf_swing_traj_aobj)
        if ssi[1] and 'link6' in ssi[1] and (time >= time_begin+self.dt/2 and time < time_end-self.dt/2):
            if time - time_begin < self.dt:
                lf_swing_traj_qp = ssi[7][int((time - time_begin)/self.dt)]
            if time_end - time < self.dt:
                lf_swing_traj_qn = ssi[7][int((time - time_begin)/self.dt)]
            lf_swing_traj_qobj = ssi[7][int((time - time_begin)/self.dt)]
            lf_swing_traj_qn = ssi[7][int((time - time_begin)/self.dt)+1]
            lf_swing_traj_qp = ssi[7][int((time - time_begin)/self.dt)-1]
            lf_swing_traj_vobj = (lf_swing_traj_qn - lf_swing_traj_qobj)/2
            lf_swing_traj_vp = (lf_swing_traj_qobj - lf_swing_traj_qp)/2
            lf_swing_traj_aobj = (lf_swing_traj_vobj - lf_swing_traj_vp)/2
            # lf_swing_traj_point.append('swing')
            lf_state = 'swing'
            lf_swing_traj_point.append(lf_swing_traj_qobj)
            lf_swing_traj_point.append(lf_swing_traj_vobj)
            lf_swing_traj_point.append(lf_swing_traj_aobj)
        elif ssi[1] and 'link6' in ssi[1] and time < time_begin+self.dt/2:
            lf_swing_traj_qobj = ssi[7][0]
            lf_swing_traj_vobj = np.zeros(3)
            lf_swing_traj_aobj = np.zeros(3)
            lf_state = 'fixed'
            lf_swing_traj_point.append(lf_swing_traj_qobj)
            lf_swing_traj_point.append(lf_swing_traj_vobj)
            lf_swing_traj_point.append(lf_swing_traj_aobj)
        elif ssi[1] and 'link6' in ssi[1] and time >= time_end-self.dt/2:
            lf_swing_traj_qobj = ssi[7][-1]
            lf_swing_traj_vobj = np.zeros(3)
            lf_swing_traj_aobj = np.zeros(3)
            lf_state = 'fixed'
            lf_swing_traj_point.append(lf_swing_traj_qobj)
            lf_swing_traj_point.append(lf_swing_traj_vobj)
            lf_swing_traj_point.append(lf_swing_traj_aobj)
        else:
            lf_swing_traj_qobj = ssi[7]
            lf_swing_traj_vobj = np.zeros(3)
            lf_swing_traj_aobj = np.zeros(3)
            # lf_swing_traj_point.append('fixed')
            lf_state = 'fixed'
            lf_swing_traj_point.append(lf_swing_traj_qobj)
            lf_swing_traj_point.append(lf_swing_traj_vobj)
            lf_swing_traj_point.append(lf_swing_traj_aobj)
        return lf_state,rf_state,lf_swing_traj_point, rf_swing_traj_point



if __name__ == '__main__':
    # [1,['t1','t2']]
    # 0.19,0.15,0; 0.095,-0.15,0
    # 0.49,0.15,0; 0.19,-0.15,0
    cs1 = ContactSchedule()
    ff = cs1.ff
    fb = cs1.fb
    rf1 = np.array([0.06,-0.15,0])
    lf1 = np.array([0.06,0.15,0])
    # print(rf1+lf1) # all need to be divided by 2
    # print(rf1+lf1+cs1.ff)
    # print(rf1)
    # print(rf1+cs1.ff)
    rf2 = np.array([0.46,-0.15,0])
    lf2 = np.array([0.46,0.15,0])
    # print(rf1+cs1.ff+lf2+cs1.fb)
    # print(lf2)
    # print(lf2+rf2)

    # two steps: start and stop
    # cs1.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(rf1+lf1)/2)
    # cs1.pushContactList(0.3,['t1','t2','t3','t4','t5','t6'],(rf1+lf1+ff)/2)
    # cs1.pushContactList(0.3,['t1','t2','t3','t4'],(rf1))
    # cs1.pushContactList(0.1,['t1','t2'],(rf1+ff))
    # cs1.pushContactList(0.2,['t1','t2','t5','t6'],(rf1+ff+lf2+fb)/2)
    # cs1.pushContactList(0.3,['t5','t6','t7','t8'],(lf2))
    # cs1.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(lf2+rf2)/2)
    # # print(cs1.sch)
    # cs1.printInfo()

    cs1.pushDST(rf1,lf1,np.array([0.0,0.0,0]),0.8)
    cs1.pushDS(rf1,lf1,0.3)
    cs1.pushSSOnR(rf1,lf2,1.5)
    cs1.pushDS(rf1,lf1,1.3)
    # cs1.pushDS(rf1,lf2,0.2)
    # cs1.pushSSOnL(rf2,lf2,0.5)
    # cs1.pushDS(rf2,lf2)
    cs1.printInfo()
    # print(cs1.swing_sch)
    # cs1.printSwingSch()
    cs1.N = int(cs1.finaltime/cs1.dt)
    # print('N:',cs1.N)

    # print(cs1.getCurrentsid(0))
    cx0 = np.array([0.0,0.0])
    cy0 = np.array([0.0,0.0])
    final_cx = np.array([((rf1+lf1)/2)[0],0.0])
    final_cy = np.array([((rf1+lf1)/2)[1],0.0])
    # final_cx = np.array([((rf2+lf2)/2)[0],0.0])
    # final_cy = np.array([((rf2+lf2)/2)[1],0.0])
    # print(final_cx,final_cy)
    cs1.getTraj(0,cx0,cy0,final_cx,final_cy)
    cs1.plotTraj()

    cs1.getSwingTraj()
    # cs1.printSwingSch()
    # print(len(cs1.swing_sch[2][6]))
    # print(cs1.swing_sch[2][6])
    time = 0
    for i in range(cs1.N):
        print(time,'\t',cs1.get_swing_traj_point(time))
        time += cs1.dt
    time = 0
    # for i in range(cs1.N):
    #     print(time,'\t',cs1.get_com_traj_point(time))
    #     time += cs1.dt

    print('time',time)
    print('N',cs1.N)

    input()
    # for i in range(cs1.N-1):
    #     print(cs1.getxyft(i*0.01))
    # print(len(cs1.sch))
    # print(cs1.sch[0][0])
    # print(cs1.getCurrentsid(0.2))
