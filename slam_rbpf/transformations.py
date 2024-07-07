import numpy as np
from math import cos, sin

def twoDSmartPlus(new_odom, odom_diff, type='pose'):
    """Return smart plus of two poses in order (x1 + x2)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = new_odom[2]
    R_theta1 = twoDRotation(theta1)
    # print '------ Rotation theta1:', R_theta1
    theta2 = odom_diff[2]
    sum_theta = theta2 + theta1
    p1 = new_odom[0:2]
    p2 = odom_diff[0:2]
    # print 'p2:', p2
    trans_of_u = p1 + np.dot(R_theta1, p2)
    # print '------ transition of u:', trans_of_u-p1
    if type == 'pose':
        return np.array([trans_of_u[0], trans_of_u[1], sum_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(sum_theta)
    return np.array([[rot_of_u[0,0],rot_of_u[0,1],trans_of_u[0]],\
                     [rot_of_u[1,0],rot_of_u[1,1],trans_of_u[1]],\
                     [0            ,   0         ,   1]])

def twoDSmartMinus(new_odom, old_odom, type='pose'):
    """
    Return smart minus of two poses in order (x2 - x1)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = old_odom[2]
    R_theta1 = twoDRotation(theta1)
    theta2 = new_odom[2]
    delta_theta = theta2 - theta1
    p1 = old_odom[0:2]
    p2 = new_odom[0:2]
    trans_of_u = np.dot(R_theta1.T, (p2 - p1))
    if type == 'pose':
        return np.array([trans_of_u[0], trans_of_u[1], delta_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(delta_theta)
    return np.array([[rot_of_u[0,0],rot_of_u[0,1],trans_of_u[0]],\
                     [rot_of_u[1,0],rot_of_u[1,1],trans_of_u[1]],\
                     [0            ,   0         ,   1]])

def twoDRotation(theta):
    """Return rotation matrix of rotation in 2D by theta"""
    return np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])

def twoDTransformation(x,y,theta):
    """Return transformation matrix of rotation in 2D by theta combining with a translation
    (x, y)"""
    return np.array([[cos(theta), -sin(theta), x],[sin(theta), cos(theta), y],[0,0,1]])

def _world_to_map(world_x, world_y, MAP):
    '''
        Converts x,y from meters to map coods
    '''
    map_x = np.ceil((world_x - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    map_y = np.ceil((world_y - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    return map_x, map_y

def _map_to_world(map_x, map_y, MAP):
    '''
        Converts x,y from map to world
    '''
    world_x = ((map_x + 1) * MAP['res']) + MAP['xmin']
    world_y = ((map_y + 1) * MAP['res']) + MAP['ymin']
    return np.vstack((world_x, world_y))

def _bresenham2D(sx, sy, ex, ey, MAP):
    sx = int(np.round(sx))
    sy = int(np.round(sy))
    ex = int(np.round(ex))
    ey = int(np.round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 
    
    if dy == 0:
        q = np.zeros((dx+1,1))
    else:      
        arange = np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1, -dy)  
        mod = np.mod(arange,dx)
        diff = np.diff(mod)  
        great =  np.greater_equal(diff,0) 
        q = np.append(0, great)
    
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
        
    x_valid = np.logical_and(x >= 0, x < MAP['sizex'])
    y_valid = np.logical_and(y >= 0, y < MAP['sizey'])
    cell_valid = np.logical_and(x_valid, y_valid)
    x = x[cell_valid]
    y = y[cell_valid]
    
    return np.vstack((x,y)).astype(int)