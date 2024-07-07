############################################
#       University of Laurentian
#     Authors: Guangtian GONG
#   Rao-Blackwellized Paricle Filter SLAM
#             RBPF SLAM Class
############################################

import numpy as np
from . import utils as utils
from . import scan_matching as matching
from . import update_models as models
from . import transformations as tf
from math import cos as cos
from math import sin as sin
import matplotlib.pyplot as plt
import time

class Particle():
    
    def __init__(self, map_dimension, map_resolution, num_p, delta = 0.005, sample_size = 10):

        self._init_map(map_dimension, map_resolution)              
        self.weight_ = 1 / num_p #inital value???
        self.weight_factor_ = None
        self.delta = delta
        self.sample_size = sample_size
        self.trajectory_ = np.zeros((3,1),dtype=np.float64) 
        self.traj_indices_ = np.zeros((2,1)).astype(int)
        
        self.log_p_true_ = np.log(9)
        self.log_p_false_ = np.log(1.0/9.0)
        self.p_thresh_ = 0.6
        self.logodd_thresh_ = np.log(self.p_thresh_/(1-self.p_thresh_))
    
    def _init_map(self, map_dimension=35, map_resolution=0.05):
        '''
        map_dimension: map dimention from origin to border
        map_resolution: distance between two grid cells (meters)
        '''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -map_dimension  #meters
        MAP['ymin']  = -map_dimension
        MAP['xmax']  =  map_dimension
        MAP['ymax']  =  map_dimension
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64) #DATA TYPE: char or int8
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'], self.MAP_['sizey']), dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'], self.MAP_['sizey']), dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'], self.MAP_['sizey']), dtype = np.uint64)
       
    def _build_first_map(self, _data):
        '''
        Builds initial map using lidar scan 'z' at initial pose
        '''
        scan = _data.lidar_data
        obstacle = scan < _data.lidar_max_
        world_x, world_y = _data._polar_to_cartesian(scan, None)
        map_x, map_y = tf._world_to_map(world_x, world_y, self.MAP_)
        r_map_x, r_map_y = tf._world_to_map(0, 0, self.MAP_)
        #print(f"obstacle = {obstacle}")
        for ray_num in range(len(scan)):
            cells_x, cells_y = tf._bresenham2D(r_map_x, r_map_y, map_x[ray_num], map_y[ray_num], self.MAP_)
            self.log_odds_[cells_x[:-1], cells_y[:-1]] += self.log_p_false_
            if obstacle[ray_num]:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_true_
            else:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_false_
        self.occu_ = 1 - (1 / (1 + np.exp(self.log_odds_)))
        self.MAP_['map'] = self.occu_ > self.p_thresh_
        
        self.traj_indices_[0], self.traj_indices_[1] = r_map_x, r_map_y
        
        occupied_map = np.where(self.log_odds_ > self.logodd_thresh_)
        self.occupied_pts_ = np.vstack((occupied_map[0], occupied_map[1]))
        
        #plt.imshow(self.MAP_['map'])
        #plt.show()
        #plt.imshow(self.occu_)
        #plt.show()
        
    def _predict(self, old_odom, new_odom, mov_cov):
        '''
        Applies motion model on last pose in 'trajectory'
        Returns predicted pose
        '''
        #print("Predict enter")
        old_pose = self.trajectory_[:,-1]       
        odom_diff = tf.twoDSmartMinus(new_odom, old_odom)     
        noise = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).flatten()

        pred_pose = tf.twoDSmartPlus(old_pose, odom_diff)
        pred_with_noise = tf.twoDSmartPlus(pred_pose, noise)
        #print("Predict exit")
        return pred_pose, pred_with_noise
        

    def _scan_matching(self, init_scan, prev_scan, cur_scan, cur_pose):
        '''
        Performs scan matching and returns (true,scan matched pose) or (false,None)
        '''
        #print("_scan_matching enter----")

        curr_scan_data = cur_scan.lidar_data - init_scan.lidar_data
        prev_scan_data = prev_scan.lidar_data - init_scan.lidar_data
        
        curr_coordinates = utils.dist_to_xy(curr_scan_data, cur_scan.lidar_angles_)
        prev_coordinates = utils.dist_to_xy(prev_scan_data, prev_scan.lidar_angles_)
        prev_pose = self.trajectory_[:, -1]
        flag, updated_pose = matching.scan_matcher(
            prev_coordinates.copy(), prev_pose.copy(), curr_coordinates.copy(), cur_pose.copy())      
        #print(f"----_scan_matching exit with flag = {flag}")
        return flag, updated_pose
    
    
    def _sample_poses_in_interval(self, scan_match_pose):
        '''
        scan matched pose: (3,1)
        Returns list of samples (3,sample_size)
        '''
        #print("_sample_poses_in_interval enter----")

        scan_match_pose = scan_match_pose.reshape((3,1))
        
        samples = np.random.random_sample((3, self.sample_size))    #### can allocate different delta's for x,y,theta
        samples = 2 * samples * self.delta - self.delta
        samples = samples + scan_match_pose
        #print("----_sample_poses_in_interval exit")
        return samples
        
    
    def _compute_new_pose(self, data_, prev_odom, cur_odom, pose_samples):
        '''
        Computes mean, cov, weight factor from pose_samples
        Samples new_pose from gaussian and appends to trajectory
        Updates weight
        '''
        #print("_compute_new_pose enter----")

        mean = np.zeros((3,))
        variance = np.zeros((3,3))
        eta = np.zeros(pose_samples.shape[1])
        pose_prev = self.trajectory_[:,-1]
        
        for i in range(pose_samples.shape[1]):
            prob_measurement = models.measurement_model(data_, pose_samples[:,i], self.occupied_pts_.T, self.MAP_)
            odom_measurement = models.odometry_model(pose_prev, pose_samples[:,i], prev_odom, cur_odom)
            eta[i] = (prob_measurement) * (odom_measurement)
            mean += pose_samples[:,i] * eta[i]
        #print('Eta: ',np.sum(eta))
        if np.sum(eta) == 0:
            return np.array([np.inf, np.inf, np.inf])
        mean = mean / np.sum(eta)
        mean = np.reshape(mean,(3,1))
        for i in range(pose_samples.shape[1]):
            variance += (pose_samples[:,i] - mean)@((pose_samples[:,i] - mean).T) * eta[i]
        variance = variance / np.sum(eta)   
        new_pose = np.random.multivariate_normal(mean.flatten(), variance)       
        self.weight_ = self.weight_ * np.sum(eta)
        #print("----_compute_new_pose exit")
        return new_pose
            
    def _update_map(self, data_, pose):
        '''
            Updates map with lidar scan z for last pose in trajectory
        '''
        update_time_1 = time.time()
        scan = data_.lidar_data
        obstacle = scan < data_.lidar_max_
        world_x, world_y = data_._polar_to_cartesian(scan, pose)
        map_x, map_y = tf._world_to_map(world_x, world_y, self.MAP_)
        r_map_x, r_map_y = tf._world_to_map(pose[0], pose[1], self.MAP_)
        update_time_2 = time.time()
        print(f"----------_update_map (1)------{(update_time_2 - update_time_1):.5f} seconds")
        for ray_num in range(len(scan)):
            cells_x, cells_y = tf._bresenham2D(r_map_x, r_map_y, map_x[ray_num], map_y[ray_num], self.MAP_)
            if cells_x.shape[0] > 0:
                self.log_odds_[cells_x[:-1], cells_y[:-1]] += self.log_p_false_
                if obstacle[ray_num]:
                    self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_true_
                else:
                    self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_false_
        self.occu_ = 1 - (1 / (1 + np.exp(self.log_odds_)))
        self.MAP_['map'] = self.occu_ > self.p_thresh_
        update_time_3 = time.time()
        print(f"----------_update_map (2)------{(update_time_3 - update_time_2):.5f} seconds")
        self.traj_indices_ = np.append(self.traj_indices_, np.array([[r_map_x],[r_map_y]]), 1)
        self.trajectory_ = np.append(self.trajectory_, np.reshape(pose,(3,1)), 1)
        occupied_map = np.where(self.log_odds_ > self.logodd_thresh_)
        self.occupied_pts_ = np.vstack((occupied_map[0], occupied_map[1]))
