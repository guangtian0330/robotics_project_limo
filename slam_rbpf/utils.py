import numpy as np
import math
from math import cos,sin

time_index_0 = 0

def transformation_scans(scan_data, pose_data, debug = False):
    """
    Parameters
    ----------
    scan_data : Scan at time t-1
        Shape: (n,2)
    pose_data : change in pose
        Shape: (3,).

    Returns
    -------
    trans_scan: transformed scan
         shape: (n,2) 
    """
    R = twoDRotation(pose_data[2])
    pose_data = pose_data.flatten()
    if debug:
      print('R:',R)
    # if scan_data.shape != (1080,2):
    #     print(scan_data.shape)
    # assert scan_data.shape == (1080,2)
    # assert pose_data.shape == (3,)
    if scan_data.shape[1] != 2:
        scan_data = scan_data.T

    trans_scan = np.dot(R, scan_data.T).T 
    trans_scan += pose_data[:2]

    assert trans_scan.shape == scan_data.shape    
    return trans_scan

def twoDRotation(theta):
    """Return rotation matrix of rotation in 2D by theta"""
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def twoDSmartPlus(x1,x2,type='pose'):
    """Return smart plus of two poses in order (x1 + x2)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = x1[2]
    R_theta1 = twoDRotation(theta1)
    # print '------ Rotation theta1:', R_theta1
    theta2 = x2[2]
    sum_theta = theta2 + theta1
    p1 = x1[0:2]
    p2 = x2[0:2]
    # print 'p2:', p2
    trans_of_u = p1 + np.dot(R_theta1, p2)
    # print '------ transition of u:', trans_of_u-p1
    if type=='pose':
        return np.array([trans_of_u[0], trans_of_u[1],sum_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(sum_theta)
    return np.array([[rot_of_u[0,0],rot_of_u[0,1],trans_of_u[0]],\
                     [rot_of_u[1,0],rot_of_u[1,1],trans_of_u[1]],\
                     [0            ,   0         ,   1]])

def twoDSmartMinus(x2,x1,type='pose'):
    """Return smart minus of two poses in order (x2 - x1)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = x1[2]
    R_theta1 = twoDRotation(theta1)
    theta2 = x2[2]
    delta_theta = theta2 - theta1
    p1 = x1[0:2]
    p2 = x2[0:2]
    trans_of_u = np.dot(R_theta1.T, (p2-p1))
    if type=='pose':
        return np.array([trans_of_u[0], trans_of_u[1],delta_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(delta_theta)
    return np.array([[rot_of_u[0,0], rot_of_u[0,1], trans_of_u[0]],\
                     [rot_of_u[1,0], rot_of_u[1,1], trans_of_u[1]],\
                     [0            ,   0         ,   1]])
def dist_to_xy(scan, angles):
    """
    Scan: (1080,) array of distances
    angles: (1080,) angles of the scan
    dpose: distance between the frames
    xy: (1080,2) array of cartisian cordinates
    """
    xy = np.zeros((scan.shape[0], 2))
    xy[:, 0] = np.cos(angles) * scan
    xy[:, 1] = np.sin(angles) * scan
    return xy
    
    
def update_weights(weights, correlations):
	"""Return weights based on correlation values"""
	# update weights
	log_weights = np.log(weights)
	# print '--log_weights:', log_weights
	log_weights += correlations
	# print '---- add correlations to obtain', log_weights
	# normalize log_weights
	max_log_weight = np.max(log_weights)
	# print '---- max_log_weight:', max_log_weight
	log_weights = log_weights - max_log_weight - logSumExp(log_weights,max_log_weight)
	# retreive weight values
	return np.exp(log_weights)    

def logSumExp(log_weights,max_log_weight):
	delta_log_weight = np.sum(np.exp(log_weights - max_log_weight))
	# print '--delta_log_weight:', delta_log_weight
	return np.log(delta_log_weight)
    
    