import numpy as np 
import math 

def calcOrientation_ori(pose, msg, deltaT):

    x = msg.x
    y = msg.y
    z = msg.z
    w = msg.w 

    f = 2 * (w*y - x*z)

    r = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p = 0
    if (-1 <= f <= 1):
        p = math.asin(f)
    
    y = math.atan2(2*(w*z+x*y), 1-2*(z*z+y*y))

    pose.orien = np.array([r, p, y]) # degree

    return pose

def calcOrientation(pose, msg, deltaT):

    B = np.array([[0, -msg.z*deltaT, msg.y*deltaT],
         [msg.z*deltaT, 0, -msg.x*deltaT],
         [-msg.y*deltaT, msg.x*deltaT, 0]])
    
    sigma = np.sqrt(msg.x**2 + msg.y**2 + msg.z**2)

    # if sigma < 1e-12:
    #     return pose

    pose.orien = np.dot(pose.orien, (np.eye(3) + np.sin(sigma)/(sigma)*B - (1-np.cos(sigma))/(sigma**2)*np.dot(B, B.T)).T)

    sy = math.sqrt(pose.orien[0][0]**2 + pose.orien[1][0]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(pose.orien[2][1], pose.orien[2][2])
        y = math.atan2(-pose.orien[2][0], sy)
        z = math.atan2(pose.orien[1][0], pose.orien[0][0])
    else:
        x = math.atan2(-pose.orien[1][2], pose.orien[1][1])
        y = math.atan2(-pose.orien[2][0], sy)
        z = 0
    pose.euler = np.array([x, y, z])*180./math.pi

    return pose

def calcPosition(pose, msg, deltaT, vel, gravity):

    acc_l = np.array([msg.x, msg.y, msg.z])
    acc_g = np.dot(acc_l, pose.orien.T)

    vel =  deltaT * (acc_g - gravity) / 2

    pose.pos = pose.pos + deltaT * vel
    return pose, vel