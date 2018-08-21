#coding=utf-8
import math
import time
import random
import numpy as np
import PPO_Agent
import matplotlib.pyplot as plt

try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *

    USE_PYQT5 = True
except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

    USE_PYQT5 = False

import threading
import socket
import sys
import struct
import gym

'''
这里使用的坐标系为X正方向向??? Y正方向向???
'''

# Linux下需要设置成False
# 如果在Windows下刷新频率不正常, 设置此项为True
ViewTitle = False

# 单位使用: 秒，???

# 碰撞距离, CRASH_R是障碍机器人的半???
CRASH_R = 0.16

# 飞行器速度
PLANE_V = 1.0
PLANE_ALPHA = 1
PLANE_BETA = 3

PLANE_PERCEPTION = 2.0  # 预测范围
PLANE_DGOAL = 1.0  # 防止引力过大的距???
PLANE_N = 2
PLANE_FREPFIX = 0.5  # 抑制斥力过大的参???

PLANE_TIME = 0  # 预测XX秒后障碍机器人的位置
PLANE_AI = True  # 是否使用策略
PLANE_REAL = False  # 使用M100
MAX_UAV_HEIGHT = 3.0

# 飞行器起始位???
START_X = -10.0
START_Y = -6.0

# 障碍机器人速度,起始半径
OBSTACLE_V = 0.33
OBSTACLE_R = 5

# 地面机器人速度,起始半径
ROBOT_V = 0.33
ROBOT_R = 1
ROBOT_REVERSE = 2456
NoiceWaitTime = 850 / 1000.0
Rotate45WaitTime = 2456 / 4000.0
Rotate180WaitTime = 2456 / 1000.0
ReverseInterval = 20.0
NoiseInterval = 5.0

LAND_HEIGHT = 0.1
TOUCH_HEIGHT = 0.3
TOUCH_RECOVER_HEIGHT = 0.5
UPDATE_HEIGHT_DELTA = 0.1

# 策略
LeftDangerValue = ROBOT_V * 5
RightDangerValue = ROBOT_V * 5
# 触碰所用时???
AIRotate45Time = 4.0
# 飞行器降落所用时???
AIRotate180Time = 10.0
# 捕捉目标最长时???
CatchTime = 30.0;
# 最好的下降位置
GoodLandPosDis = 1

# 网络部分
HOST = ''
PORT = 6102
BUFSIZ = 1024
ADDR = (HOST, PORT)

PORT2 = 7102
ADDR2 = (HOST, PORT2)

# 刷新速度倍数(与正常速度???
UpdateRatio = 1.0
UpdateSeconds = 0.02

# 坐标线间???20 / 行或列格??? , 必须为整???
LineInterval = 20 // 10

PI = np.pi

# 碰撞
def isCrash(a, b):
    u = CRASH_R * 2
    d2 = sum((a.pos - b.pos) ** 2)
    p = b.pos - a.pos;
    w = p.dot(a.v);
    return d2 <= u * u and w > 0


# 在附???
def isNear(a, b, u):
    d2 = sum((a.pos - b.pos) ** 2)
    return d2 <= u * u


def GetDis(a, b):
    d2 = sum((a.pos - b.pos) ** 2)
    return np.sqrt(d2)


def To360A(p):
    ra = p / PI * 180
    an = ra - int(int(ra) / 360) * 360
    return an


def To360A(p):
    ra = p / PI * 180
    an = ra - int(int(ra) / 360) * 360
    return an


class MoveObject:
    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @x.setter
    def x(self, value):
        raise Exception("Don't change x")

    @y.setter
    def y(self, value):
        raise Exception("Don't change x")

    def Crashed(self):
        # 碰到地面机器人或障碍机器???
        for r in self.Robots + self.Obstacles:
            if isCrash(self, r) and self.id != r.id:
                return True
                # 碰到飞行???
        for p in self.Planes:
            if isCrash(self, p) and self.id != p.id and p.isLand():
                return True
        return False


class Obstacle(MoveObject):
    _R = OBSTACLE_R
    _V = OBSTACLE_V
    _AngleA = _V * 1.0 / _R  # 水平加速度
    _A = _V * _V * 1.0 / _R  # 垂直加速度

    def _updatepar(self):
        self.pos = np.array([self._R * np.cos(self.angle), self._R * np.sin(self.angle)])
        self.v = np.array([self._V * np.sin(self.angle), - self._V * np.cos(self.angle)])
        self.a = np.array([- self._A * np.cos(self.angle), - self._A * np.sin(self.angle)])

    def __init__(self, id, angle):
        self.angle = angle
        self._updatepar()
        self.id = id

    def update(self, seconds):
        if not self.Crashed():
            self.angle = self.angle - seconds * self._AngleA
            self._updatepar()


STATE_RUN = 0
STATE_NOISE = 1
STATE_REVERSE = 2
STATE_TOUCH = 3


class Robot(MoveObject):
    _V = ROBOT_V
    _R = ROBOT_R

    def __init__(self, id, angle):
        self.angle = angle
        self.waitTime = 0
        self.pos = np.array([self._R * np.cos(self.angle), self._R * np.sin(self.angle)])
        self.v = np.array([self._V * np.cos(self.angle), self._V * np.sin(self.angle)])
        self.a = np.array([0.0, 0.0])
        self.noise = 0
        self.turn = 0
        self.noiseTime = 0
        self.noiseAngleV = 0.0
        self.turnTime = 0
        self.state = STATE_RUN
        self.id = id
        self.touched = False
        self.stoped = False

    def update(self, seconds):

        if np.abs(self.x) > 10 or np.abs(self.y) > 10:
            self.stoped = True
        # 出界???
        if self.stoped:
            return
        '''
        if RealUAV:
            return
        '''

        # 运动规则
        # ???80度时是不受碰撞影响的???但受Touch影响
        # 0 run | touch,reverse,noise,colli
        # 1 noise | touch, colli
        # 2 reverse | touch
        #   colli -> reverse
        # 3 touch | colli
        self.noise += seconds
        self.turn += seconds
        if self.state == STATE_RUN:
            if self.isTopTouch():
                self.TopTouch()
            elif self.turn >= ReverseInterval:
                self.turn = 0.0
                self.Reverse()
            elif self.noise >= NoiseInterval:
                self.noise = 0.0
                self.TrajectoryNoise()
            elif self.Crashed():
                self.TargetCollision()
        elif self.state == STATE_NOISE:
            if self.isTopTouch():
                self.TopTouch()
            elif self.noiseTime <= 0:
                self.TargetRun()
            elif self.Crashed():
                self.TargetCollision()
        elif self.state == STATE_REVERSE:
            if self.isTopTouch():
                self.TopTouch()
            elif self.turnTime <= 0:
                self.TargetRun()
        elif self.state == STATE_TOUCH:
            if self.turnTime <= 0:
                self.TargetRun()
            elif self.Crashed():
                self.TargetCollision()

        if (self.state == STATE_TOUCH or self.state == STATE_REVERSE) and self.turnTime > 0:
            self.angle -= 45.0 / Rotate45WaitTime * seconds / 180.0 * PI
            self.turnTime -= seconds
        if self.state == STATE_NOISE and self.noiseTime > 0:
            self.angle -= self.noiseAngleV * seconds
            self.noiseTime -= seconds

        # update parameter
        newv = np.array([0.0, 0.0])
        if self.state == STATE_RUN:
            newv = np.array([self._V * np.cos(self.angle), self._V * np.sin(self.angle)])
            self.pos += newv * 1.0 * seconds

        self.a = (newv - self.v) * 1.0 / seconds
        self.v = newv

    def isTopTouch(self):
        if self.touched:
            self.touched = False
            return True
        return False

    def TopTouch(self):
        self.state = STATE_TOUCH
        self.noiseTime = 0
        self.turnTime = Rotate45WaitTime

    def Reverse(self):
        self.state = STATE_REVERSE
        self.noiseTime = 0
        self.turnTime = Rotate180WaitTime

    def TrajectoryNoise(self):
        self.state = STATE_NOISE
        self.noiseTime = NoiceWaitTime
        u = 20.0 / 180 * PI
        k = random.random() * 2 * u - u
        self.noiseAngleV = k / NoiceWaitTime
        self.turnTime = 0

    def TargetCollision(self):
        self.Reverse()

    def TargetRun(self):
        self.state = STATE_RUN
        self.noiseTime = 0
        self.turnTime = 0


# 避障算法(http://blog.csdn.net/junshen1314/article/details/50472410)
def GetFatt(apos, opos, target):
    a = target - apos
    p = np.sqrt(np.sum(a ** 2))
    if p <= PLANE_DGOAL:
        return PLANE_ALPHA * a
    else:
        return PLANE_DGOAL * PLANE_ALPHA * a / p


def GetFrep(apos, opos, target):
    Frep = np.array([0.0, 0.0])
    a = target - apos
    p = np.sqrt(np.sum(a ** 2))
    if p < PLANE_FREPFIX:
        p = PLANE_FREPFIX
    for o in opos:
        tp = o.pos.copy()
        b = apos - tp
        g = np.sqrt(np.sum(b ** 2))
        if g > PLANE_PERCEPTION:
            continue
        Frep1 = PLANE_BETA * (1.0 / g - 1.0 / PLANE_PERCEPTION) * 1.0 / (g ** 3) * b * (p ** PLANE_N)
        Frep2 = PLANE_N / 2.0 * PLANE_BETA * ((1.0 / g - 1.0 / PLANE_PERCEPTION) ** 2) * (p ** (PLANE_N - 1)) * a / p
        Frep += Frep1 + Frep2
    return Frep


def Plan(apos, opos, target):
    '''
    a = apos - target
    dj = PLANE_ALPHA * a
    for o in opos:
        tp = copy(o.pos)
        b = apos - tp
        g = np.sqrt(np.sum(b ** 2))
        if g > PLANE_PERCEPTION:
            continue
        dj += PLANE_BETA * (1.0 / g - 1 / PLANE_PERCEPTION) * (- 1.0 / (g ** 3)) * b
    return -dj
    '''
    return GetFatt(apos, opos, target) + GetFrep(apos, opos, target)


class Plane(MoveObject):
    _V = PLANE_V
    _LandV = 0.2
    alpha = PLANE_ALPHA
    beta = PLANE_BETA
    perception = PLANE_PERCEPTION  # 感知障碍物范???
    ti = PLANE_TIME

    def __init__(self, x, y):
        self.pos = np.array([y, x])
        self.v = np.array([.0, .0])
        self.a = np.array([.0, .0])
        self.targetPos = np.array([.0, .0])
        self.lockRobot = 0
        self.crashTimes = 0.0
        self.z = 1.5
        self.oldz = self.z
        self.landed = False
        self.id = 100
        self.waitTime = 0.0
        self.touched = False
        self.yaw = 0.0
        self.height_v = 1.0
        self.height_state = 0  # -1, 0, 1

    def update(self, seconds):
        for o in self.Obstacles:
            if isNear(self, o, CRASH_R * 2):
                self.crashTimes += 1

        if self.lockRobot:
            self.targetPos = self.lockRobot.pos

        if PLANE_REAL:
            return

        if not PLANE_REAL:
            if self.waitTime > 0:
                self.waitTime -= seconds
                return

            if self.isLand():
                self.Land()
                return
            dv = Plan(self.pos, self.Obstacles, self.targetPos)
            self.v = dv / np.sqrt(np.sum(dv ** 2)) * self._V

        if self.isLand():
            self.Land()
            return

        if self.height_state > 0:
            self.z = self.z + self.height_v * seconds
            self.landed = False
            if self.z >= 1.5:
                self.z = 1.5
                self.height_state = 0
        elif self.height_state < 0:
            self.z = self.z - self.height_v * seconds
            if self.z <= 0:
                self.z = 0.0
                self.height_state = 0
                self.landed = True

        if self.z != self.oldz:
            if self.z < self.oldz and self.z < TOUCH_HEIGHT:
                if not self.touched:
                    self.touched = True
                    for r in self.Robots:
                        if isNear(self, r, CRASH_R):
                            r.touched = True  # 触碰标记
                            self.landed = False
                            #print(('TopTouch Ground Robot %d  - %f' % (r.id, To360A(r.angle))))
                            break
            elif self.z > self.oldz and self.z > TOUCH_RECOVER_HEIGHT:
                self.touched = False
        if np.abs(self.z - self.oldz) >= UPDATE_HEIGHT_DELTA:
            self.oldz = self.z

        self.pos += self.v * 1.0 * seconds

    def Land(self):
        if self.isLand():
            self.landed = False
        if self.height_state == 0:
            if self.z < 0.1:
                self.height_state = 1
            elif self.z > 1.0:
                self.height_state = -1
        """
        if self.height_state != 0:
            return
        if self.landed:
            self.landed = False
            return

        self.landed = True
        for r in self.Robots:
            if isNear(self,r,CRASH_R):
                r.touched = True # 触碰标记
                self.landed = False
                print(('TopTouch Ground Robot %d  - %f' % (r.id,To360A(r.angle))))
                break
        """

    def isLand(self):
        if PLANE_REAL:
            return self.z <= LAND_HEIGHT
        return self.landed


class SimWindow(QWidget):
    def __init__(self):
        super(SimWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(320, 140, 900, 700)
        self.setWindowTitle('SimWindow')
        self.show()

    def mousePressEvent(self, event):
        p = event.pos()

        def GetXY(w, h):
            u = 600
            x = (w - 50) * 1.0 / u * 20.0 - 10
            y = -((h - 50) * 1.0 / u * 20.0 - 10)
            return x, y

        x, y = GetXY(p.x(), p.y())
        if event.button() == Qt.RightButton:
            print("===")
            for r in self.Robots:
                print((r.pos))
                u = r.pos - np.array([x, y])
                d2 = sum(u ** 2)
                if d2 < 0.5 * 0.5:
                    self.Planes[0].lockRobot = r
                    print(("lock", r.id))
                    break
        else:
            self.Planes[0].targetPos = np.array([x, y])
            self.Planes[0].lockRobot = None

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.Planes[0].Land()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw(qp)
        qp.end()

    def draw(self, qp):

        def GetWH(x, y):
            u = 600
            h = (-y + 10) / 20.0 * u + 50
            w = (x + 10) / 20.0 * u + 50
            return w, h

        # draw
        def DrawPoint(x, y):
            w, h = GetWH(x, y)
            qp.drawPoint(w, h)

        def DrawLine(x1, y1, x2, y2):
            w1, h1 = GetWH(x1, y1)
            w2, h2 = GetWH(x2, y2)
            qp.drawLine(w1, h1, w2, h2)

        def DrawBorderLines():
            pen = QPen(Qt.gray, 2, Qt.SolidLine)
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([1, 4])
            qp.setPen(pen)

            # 画网???
            for r in range(-10, 10, LineInterval):
                DrawLine(-10, r, 10, r)
                DrawLine(r, -10, r, 10)

            pen = QPen(Qt.green, 4, Qt.SolidLine)
            qp.setPen(pen)
            DrawLine(-10, 10, -10, -10)

            pen = QPen(Qt.red, 4, Qt.SolidLine)
            qp.setPen(pen)
            DrawLine(10, 10, 10, -10)

            pen = QPen(Qt.gray, 4, Qt.SolidLine)
            qp.setPen(pen)
            DrawLine(-10, 10, 10, 10)
            DrawLine(-10, -10, 10, -10)

        DrawBorderLines()

        pen = QPen(Qt.red, 6, Qt.SolidLine)
        qp.setPen(pen)
        for o in self.Obstacles:
            DrawPoint(o.x, o.y)

        pen = QPen(Qt.green, 6, Qt.SolidLine)
        qp.setPen(pen)
        for r in self.Robots:
            DrawPoint(r.x, r.y)

        Fatt = 0
        Frep = 0
        for p in self.Planes:
            # 绘制无人???
            pen = QPen(Qt.black, 6, Qt.SolidLine)
            qp.setPen(pen)
            DrawPoint(p.x, p.y)
            pen = QPen(Qt.black, 1, Qt.SolidLine)
            qp.setPen(pen)

            # 注意yaw计算的方???
            # yawL = 0.6
            # DrawLine(p.x, p.y, p.x + np.cos(p.yaw) * yawL, p.y - np.sin(p.yaw) * yawL)

            # 绘制引力斥力
            Fatt = GetFatt(p.pos, self.Obstacles, p.targetPos)
            Frep = GetFrep(p.pos, self.Obstacles, p.targetPos)
            Latt = Fatt + p.pos
            Lrep = Frep + p.pos
            L = p.pos + Fatt + Frep
            '''
            #att
            pen = QPen(Qt.green, 4 , Qt.SolidLine)
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([1, 2])
            qp.setPen(pen)
            DrawLine(p.x, p.y, Latt[0], Latt[1]) 
            #rep
            pen = QPen(Qt.red, 4 , Qt.SolidLine)
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([1, 2])
            qp.setPen(pen)
            DrawLine(p.x, p.y, Lrep[0], Lrep[1]) 
            #F
            pen = QPen(Qt.cyan, 4 , Qt.SolidLine)
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([1, 2])
            qp.setPen(pen)
            DrawLine(p.x, p.y, L[0], L[1]) 
            '''

        for p in self.Planes:
            pen = QPen(Qt.blue, 4, Qt.DashDotLine)
            qp.setPen(pen)
            DrawPoint(p.targetPos[0], p.targetPos[1])

        lineWidth = 100
        pen = QPen(Qt.red, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(732, 50, 732 + lineWidth, 50)
        pen = QPen(Qt.blue, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(732, 650, 732 + lineWidth, 650)
        pen = QPen(Qt.black, 4, Qt.DashDotLine)
        pen.setDashPattern([2, 2])
        qp.setPen(pen)
        hy = 650 - 600 * self.Planes[0].z / MAX_UAV_HEIGHT
        qp.drawLine(732, hy, 732 + lineWidth, hy)
        qp.drawText(720, 30, "Height: %.2f" % (self.Planes[0].z))

        success = 0
        fail = 0
        for r in self.Robots:
            if r.stoped:
                if r.y < -10 and r.x > -10 and r.x < 10:
                    success += 1
                else:
                    fail += 1

        if success + fail >= 10:
            self.GameOver[0] = True

        qp.setPen(Qt.black)
        qp.drawText(10, 15, "Success %d, Fail %d, crash: %d" % (success, fail, self.Planes[0].crashTimes))
        # qp.drawText(10,30,"Fatt(%.2f, %.2f), Frep(%.2f, %.2f)" % (Fatt[0], Fatt[1], Frep[0], Frep[1]))
        qp.drawText(500, 15, "UseTime: %.2f" % (self.env.useTime))
        if self.Planes[0].isLand():
            qp.drawText(10, 30, "Land")
        qp.setPen(Qt.red)
        if self.Planes[0].landed:
            qp.drawText(60, 30, "ToLand")


class IARCSimEnv(gym.Env):
    def __init__(self):
        super(IARCSimEnv, self).__init__()
        self.reset()

    def reset(self):
        self.Obstacles = [Obstacle(100 + i, PI / 2 * i) for i in range(4)]
        self.Robots = [Robot(i, 2 * PI / 10 * i) for i in range(10)]
        self.Planes = [Plane(START_X, START_Y)]
        self.old_target = 5
        self.time_interval = 20
        self.epsilon = 0.6
        self.success_robot = []
        self.fail_robot = []
        for e in self.Obstacles + self.Robots + self.Planes:
            e.Obstacles = self.Obstacles
            e.Robots = self.Robots
            e.Planes = self.Planes
        self.escape_time = 0.0
        self.sim_time = 0.0
        self.useTime = 0.0
        self.GameOver = [False]
        if hasattr(self, 'simWindow'):
            self.simWindow.Obstacles = self.Obstacles
            self.simWindow.Robots = self.Robots
            self.simWindow.Planes = self.Planes
            self.simWindow.GameOver = self.GameOver
            self.simWindow.env = self

        return self.get_state()

    def step(self, a):
        dt = 0.01
        update_times = 1
        calcTime = self.useTime % 20
        target = 7#a / 2
        action = 0#a % 2
        self.Planes[0].lockRobot = self.Robots[target]
        reward = 0 
        old_AvgPos = []
        for eachRobots in self.Robots:
            eachAvgPos = eachRobots.pos + eachRobots.v * 1.0 * (10.0 - calcTime)
            old_AvgPos.append(eachAvgPos)
            
        if action == 0: # TopTouch
            if isNear(self.Planes[0], self.Robots[target], CRASH_R):
                # print("Plane Rotates45 Robot %d from angle %f" % (r.id, To360A(r.angle)))
                self.Planes[0].Land()
            if self.Planes[0].touched and self.Planes[0].height_state <= 0:
                self.Planes[0].height_state = 1
                #reward = 10
                self.Planes[0].waitTime = AIRotate45Time
                update_times = int(math.ceil(Rotate45WaitTime / dt))

        elif action == 1: # Collision
            if isNear(self.Planes[0], self.Robots[target], CRASH_R):
                if self.Planes[0].waitTime <= 0 and self.Planes[0].height_state == 0:
                    self.Robots[target].Reverse()
                    self.Planes[0].waitTime = AIRotate180Time
                    #reward = 10
                    update_times = int(math.ceil(Rotate180WaitTime / dt))
                    #print("Plane Rotates180 Robot %d from angle %f" % (r.id, To360A(r.angle)))
        #if update_times > 1:
        #    print("action",action,"  update time",update_times)
        #    print("START 7", Robots[7].pos)
        for _ in range(update_times):
            for o in self.Obstacles:
                pass#o.update(dt)
            for r in self.Robots:
                r.update(dt)
            for p in self.Planes:
                p.update(dt)

        next_state = self.get_state()

        if self.Planes[0].waitTime <= 0 and not self.Planes[0].touched:
            if self.old_target != target:
                reward = -3
            else:
                reward = -1

        self.useTime += dt * update_times
        newCalcTime = self.useTime % 20
        next_AvgPos = []
        for eachRobots in self.Robots:
            eachAvgPos = eachRobots.pos + eachRobots.v * 1.0 * (10.0 - newCalcTime)
            next_AvgPos.append(eachAvgPos)    
        next_AvgPosArray = np.array(next_AvgPos)
        old_AvgPosArray = np.array(old_AvgPos)
        allDeltaPos = next_AvgPosArray - old_AvgPosArray
        allDeltaX = np.sum((allDeltaPos),axis=0)[0]
        print("delta Avg X",allDeltaPos[7][0])

        if abs(allDeltaX) > self.epsilon:
            reward += -  allDeltaX
        reward += -  allDeltaPos[target][0]    
        #print(- 100.0 * allDeltaPos[target][0])
                # if self.Robots[target].x < -10 and target not in self.success_robot:
                #     reward = 10000
                #     self.success_robot.append(target)
                #     print("success",target)

            #for temp_robot in self.Robots :
            #    if temp_robot.x < -10 and temp_robot not in self.success_robot:
            #        reward += 20
            #        self.success_robot.append(temp_robot)
            #        print("success")
            #    elif temp_robot.x > 10 and temp_robot not in self.fail_robot:
            #        reward += -1000
            #        self.fail_robot.append(temp_robot)
            #        print("fail")

        if r in self.success_robot or r in self.fail_robot:
            reward += -20


        self.old_target = target
        return next_state, reward, self.GameOver[0], self.useTime, {}

    def render(self):
        if hasattr(self, 'simWindow'):
            self.simWindow.update()

    def get_state(self):
        return np.hstack(
            # [[r.x, r.y, r.angle] for r in self.Robots] +
            # [[r.x, r.y, r.angle] for r in self.Obstacles] +
            # [[p.x, p.y, p.z] for p in self.Planes]
            [[GetDis(r,self.Planes[0])] for r in self.Robots] +
            [[(r.x - (-10))] for r in self.Robots] +
        [[r.angle] for r in self.Robots] +
        #[[GetDis(r,self.Planes[0])] for r in self.Obstacles] +
        #[[r.angle] for r in self.Obstacles] +
        [[(self.useTime % 20)]]
        )


class StrategyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.ppo = PPO_Agent.PPO()
        self.episode_n = PPO_Agent.EP_MAX
        self.episode_l = PPO_Agent.EP_LEN
        self.batch = PPO_Agent.BATCH
        self.gamma = PPO_Agent.GAMMA

    def run(self):
        all_ep_r = []
        for ep in range(self.episode_n):
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(self.episode_l):  # in one episode
                #if ep % 10 == 0:
                env.render()
                a = self.ppo.choose_action(s)
                s_, r, done, currentTime, _ = env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)  # normalize reward, find it useful
                s = s_
                ep_r += r
                # update ppo
                if (t + 1) % self.batch == 0 or t == self.episode_l - 1 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.ppo.get_v(s_)

                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.hstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.ppo.update(bs, ba, br)

                    if done:
                        break
            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)  # moving average操作使得奖励变化曲线更为平滑
            print(
                'Ep: %i' % ep,
                "|Ep_r: %i" % ep_r,
                ("|Lam: %.4f" % PPO_Agent.METHOD['lam']) if PPO_Agent.METHOD['name'] == 'kl_pen' else '',
            )

        plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.show()


if __name__ == '__main__':
    env = IARCSimEnv()
    t = StrategyThread()
    t.setDaemon(True)
    t.start()

    app = QApplication(sys.argv)
    env.simWindow = SimWindow()
    env.reset()
    sys.exit(app.exec_())
