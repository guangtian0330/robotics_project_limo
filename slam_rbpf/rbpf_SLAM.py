# -*- coding: utf-8 -*-

############################################
#       University of Laurentian
#     Computational Science
#     Authors: Guangtian Gong
#   Rao-Blackwellized Paricle Filter SLAM
#             RBPF SLAM Class
############################################

import numpy as np
from .rbpf_particle import Particle
import matplotlib.pyplot as plt
import cv2
from . import update_models as models
import copy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
import time

def_zero_threshold = 1e-15


class SLAM():
    def __init__(self, mov_cov, num_p = 10, map_resolution = 0.05, map_dimension = 50, Neff_thresh = 3):
        self.num_p_ = num_p
        self.Neff_ = 0
        self.Neff_thresh_ = Neff_thresh
        self.weights_ = np.ones(num_p) / num_p
        self.mov_cov_ = mov_cov
        self.lidar_data = None
        self.odom_data_list = []
        self.particles_ = []
        self.grid_msg = None
        self.map_img = None
        for i in range(self.num_p_):
            self.particles_.append(Particle(map_dimension, map_resolution, num_p))        
        self.Neff = 1 / np.sum(self.weights_ ** 2)
        print(f" ----------------------init Neff = {self.Neff}-------------------------")

    def add_lidar_data(self, data):
        self.lidar_data = data

    def add_odo_data(self, data):
        self.odom_data_list.append(data)

    def check_lidar_valid(self):
        return self.lidar_data is not None

    def clear_data(self):
        self.lidar_data = None
        self.odom_data_list = []

    """
    Get the odometry data at a specified time point.
    """
    def get_odom_at_time(self, lidar_data):
        if not self.odom_data_list:
            return np.array([])
        odom_idx = np.argmin([np.abs(od.odom_time - lidar_data.scan_time) for od in self.odom_data_list])
        return np.array([self.odom_data_list[odom_idx].x, self.odom_data_list[odom_idx].y, self.odom_data_list[odom_idx].theta])
        
    def _resample(self):
        c = self.weights_[0]
        j = 0
        u = np.random.uniform(0, 1.0 / self.num_p_)
        new_particles = []
        for k in range(self.num_p_):
            beta = u + float(k) / self.num_p_
            while beta > c and j < self.weights_.size - 1:
                j += 1
                c += self.weights_[j]
            print(f"_resample ----- Choosed particle[{j}] with weight value = {self.weights_[j]}, c = {c}, beta = {beta}")
            new_particles.append(copy.deepcopy(self.particles_[j]))
        
        self.weights_ = np.ones(self.num_p_) / self.num_p_     
        self.particles_ = new_particles
        self.Neff = 1 / np.sum(self.weights_ ** 2)
        print(f"_resample --EXIT--- self.Neff ={self.Neff}")

    def _init_map_for_particles(self):
        self.init_scan = self.lidar_data
        self.prev_scan = self.lidar_data
        self.prev_odom = self.get_odom_at_time(self.prev_scan)
        for p in self.particles_:
            p._build_first_map(self.init_scan)

    def _run_slam(self):
        '''
            Performs SLAM
        '''
        #run_start_time = time.time()
        #for index in range(t0, t_end):
        #start_time = time.time()       
        cur_scan = self.lidar_data
        cur_odom = self.get_odom_at_time(cur_scan)
        for i, p in enumerate(self.particles_):
            # predict with motion model
            #predict_time = time.time()
            print(f"-----------------PARTICLE{i}--------------------------------------------------------------")
            pred_pose, pred_with_noise = p._predict(self.prev_odom, cur_odom, self.mov_cov_)
            #sucess, scan_match_pose =  p._scan_matching(self.init_scan, self.prev_scan, cur_scan, pred_pose)
            #print(f"|----_run_slam pred_pose = {pred_pose}, pred_with_noise = {pred_with_noise}, scan_match_pose = {scan_match_pose}, sucess={sucess}")
            if p.occupied_pts_.T.size == 0:
                continue
            # use motion model for pose estimate
            est_pose = pred_with_noise
            measure = models.measurement_model(cur_scan, pred_with_noise, p.occupied_pts_.T, p.MAP_)
            print(f"|-----measurement_model, current weight{i} is {self.weights_[i]}, measure = {measure}----------------")
            if measure > def_zero_threshold :
                p.weight_ = p.weight_ * measure
            else :
                print(f"|-----no change for particle{i}, and current weight is {self.weights_[i]}, measure = {measure}-----")
            """
            if not sucess:
                if p.occupied_pts_.T.size == 0:
                    continue
                # use motion model for pose estimate
                est_pose = pred_with_noise
                measure = models.measurement_model(cur_scan, pred_with_noise, p.occupied_pts_.T, p.MAP_)
                print(f"|-----measurement_model, current weight{i} is {self.weights_[i]}, measure = {measure}----------------")
                if measure > def_zero_threshold :
                    p.weight_ = p.weight_ * measure
                else :
                    print(f"|-----no change for particle{i}, and current weight is {self.weights_[i]}, measure = {measure}-----")
            else:
                # sample around scan match pose
                sample_poses = p._sample_poses_in_interval(scan_match_pose)
                est_pose = p._compute_new_pose(cur_scan, self.prev_odom, cur_odom, sample_poses)
                if np.isinf(est_pose).any():
                    print(f"|-----no change for particle{i}, and current weight is {self.weights_[i]}-------")
                    continue
            """
            self.weights_[i] = p.weight_
            #update_time = time.time()
            p._update_map(cur_scan, est_pose)
            #update_elapsed = time.time() - update_time
            print(f"|-----_update_map  finished ------ self.weights_[{i}] = {self.weights_[i]}")
            print(f"----------------------------------------------------------------------------------------")
        self.Neff = 1 / np.sum(self.weights_ ** 2)
        print(f"Calculating the neff value after running all particles.------ self.Neff = {self.Neff}")
        if self.Neff < self.Neff_thresh_:
            self._resample()
        self.prev_scan = cur_scan
        self.prev_odom = cur_odom
        #slam_time = time.time()
        #print(f"---_run_slam---{(slam_time - run_start_time):.5f} seconds-----number of particles = {self.num_p_}-----")
        # Generate the map based on the partichle with the larges weight.
        #combined_log_odds, combined_traj = self._combine_maps(5)
        #combine_time = time.time()
        print(f"----The updated pose is {self.particles_[np.argmax(self.weights_)].traj_indices_[:,-1]} from particle[{np.argmax(self.weights_)}]---------")
        self._gen_map(self.particles_[np.argmax(self.weights_)])
        
    def _combine_maps(self, n = 5):
        '''
            Combines maps of the top n particles into a single global map and trajectory
        '''
        # Acquire the n particles with the largest weights.
        top_n_indices = np.argsort(self.weights_)[-n:]
        #print(f"the weight list:{self.weights_}")
        #print(f"the index list of the top 5 weights:{top_n_indices}")

        combined_log_odds = np.zeros_like(self.particles_[0].log_odds_)
        combined_traj = []

        for i in top_n_indices:
            p = self.particles_[i]
            combined_log_odds += p.log_odds_
            positive_indices = np.where(p.log_odds_ > 0)
            positive_values = p.log_odds_[positive_indices]
            #print(f"odds of particle{i} = {positive_values}")
            combined_traj.append(p.traj_indices_)
        combined_log_odds /= n
        combined_traj = np.concatenate(combined_traj, axis=1)
        return combined_log_odds, combined_traj


    def _gen_map(self, particle):
        '''
            Generates and publishes the combined map with trajectory to the ROS topic
        '''
        #gen_init_time1 = time.time()
        log_odds = particle.log_odds_
        logodd_thresh = particle.logodd_thresh_
        traj = particle.traj_indices_
        print(f"THE WHOLE trajectory should be: \n{traj}")
        MAP = particle.MAP_
        # Generate the visualization map
        MAP_2_display = 255 * np.ones((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
        MAP_2_display[log_odds > logodd_thresh, :] = [0, 0, 0]
        MAP['xmin']
        valid_x = (traj[0] >= 0) & (traj[0] < MAP['sizex'])
        valid_y = (traj[1] >= 0) & (traj[1] < MAP['sizey'])
        valid_indices = valid_x & valid_y   
        #MAP_2_display[abs(log_odds) < 1e-1, :] = [150, 150, 150]
        x_indices = traj[0][valid_indices]
        y_indices = traj[1][valid_indices]
        y_indices = MAP['sizey'] - 1 - y_indices
        MAP_2_display[y_indices, x_indices] = [70, 70, 228]
        # Save the visualization map
        self.map_img = cv2.resize(MAP_2_display, (700, 700))
        #indices = np.where(log_odds > 0.2)
        #xy_coordinates = list(zip(indices[1], indices[0]))  # 注意交换以适应 (x, y) 格式
        #print(f"Coordinates (x, y) where log_odds > logodd_thresh:{xy_coordinates}, odd = {logodd_thresh}")
        #print(f"traj = {traj}")
        #gen_init_time2 = time.time()
        #print(f"---_gen_map grid map---{(gen_init_time2 - gen_init_time1):.5f} seconds--")
        """
        occupancy_grid = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)
        # wall_indices = np.where(log_odds > logodd_thresh)
        # unexplored_indices = np.where(abs(log_odds) < 1e-1)
        occupancy_grid[log_odds > logodd_thresh] = 100
        occupancy_grid[abs(log_odds) < 1e-1] = -1

        if (np.all(traj[0] < occupancy_grid.shape[0]) and 
            np.all(traj[1] < occupancy_grid.shape[1])):
            occupancy_grid[traj] = 50
        traj_time = time.time()
        print(f"---update traj_time grid map---{(traj_time - gen_init_time2):.5f} seconds--")
        #print(f"wall_indices = {wall_indices}")
        #print(f"unexplored_indices = {unexplored_indices}")
        # Create an OccupancyGrid message
        self.grid_msg = OccupancyGrid()
        self.grid_msg.info = MapMetaData()
        self.grid_msg.info.resolution = MAP['res']
        self.grid_msg.info.width = MAP['sizex']
        self.grid_msg.info.height = MAP['sizey']
        self.grid_msg.info.origin.position.x = float(MAP['xmin'])
        self.grid_msg.info.origin.position.y = float(MAP['ymin'])
        self.grid_msg.info.origin.orientation.w = 1.0
        fillin_data = time.time()
        print(f"---fill in grid message1---{(fillin_data - traj_time):.5f} seconds--")
        #self.grid_msg.data = occupancy_grid.flatten().tolist()
        self.grid_msg.data = occupancy_grid
        
        #print(type(occupancy_grid.flatten()[0]))
        #print(type(list(occupancy_grid.ravel())[0]))
        grid_init_time = time.time()
        print(f"---fill in grid message2---{(grid_init_time - fillin_data):.5f} seconds--")
        """


    """
    def _save_map(self, particle, t, p_num):
        MAP = self._gen_map(particle)
        file_name = 'logs/New folder/t_'+ str(t)+'_p_'+str(p_num)+'.png'
        cv2.imwrite(file_name, MAP)
    """