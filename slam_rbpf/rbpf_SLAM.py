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


class SLAM():
    def __init__(self, mov_cov, num_p = 15, map_resolution = 0.05, map_dimension = 50, Neff_thresh = 0.5):
        self.num_p_ = num_p
        self.Neff_ = 0
        self.Neff_thresh_ = Neff_thresh
        self.weights_ = np.ones(num_p)/num_p
        self.mov_cov_ = mov_cov
        self.lidar_data_list = []
        self.odom_data_list = []
        self.particles_ = []
        for i in range(self.num_p_):
            self.particles_.append(Particle(map_dimension, map_resolution, num_p))        

    def add_lidar_data(self, data):
        self.lidar_data_list.append(data)

    def add_odo_data(self, data):
        self.odom_data_list.append(data)

    def get_shorter_length(self):
        lidar_length = len(self.lidar_data_list)
        odom_length = len(self.odom_data_list)
        return min(lidar_length, odom_length)

    def clear_data(self):
        self.lidar_data_list = []
        self.odom_data_list = []

    """
    Get the odometry data at a specified time point.
    """
    def get_odom_at_time(self, index):
        if not self.odom_data_list:
            return np.array([])
        lidar_data = self.lidar_data_list[index]
        odom_idx = np.argmin([np.abs(od.odom_time - lidar_data.scan_time) for od in self.odom_data_list])
        return np.array([self.odom_data_list[odom_idx].x, self.odom_data_list[odom_idx].y, self.odom_data_list[odom_idx].theta])
        
    def _resample(self):
        c = self.weights_[0]
        j = 0
        u = np.random.uniform(0,1.0/self.num_p_)
        new_particles = []
        for k in range(self.num_p_):
            beta = u + float(k)/self.num_p_
            while beta > c:
                j += 1
                c += self.weights_[j]
            new_particles.append(copy.deepcopy(self.particles_[j]))
        
        self.weights_ = np.ones(self.num_p_)/self.num_p_     
        self.particles_ = new_particles

    def _init_map_for_particles(self):
        self.init_scan = self.lidar_data_list[0]
        self.prev_scan = self.lidar_data_list[0]
        self.prev_odom = self.get_odom_at_time(0)
        for p in self.particles_:
            p._build_first_map(self.init_scan)

    def _run_slam(self, t0, t_end):
        '''
            Performs SLAM
        '''
        #start_time = time.time()  # 记录函数开始时间
        for index in range(t0, t_end):
            start_time = time.time()  # 记录函数开始时间               
            cur_scan = self.lidar_data_list[index]
            cur_odom = self.get_odom_at_time(index)
            for i, p in enumerate(self.particles_):
                # predict with motion model
                predict_time = time.time()  # 记录函数开始时间 
                pred_pose, pred_with_noise = p._predict(self.prev_odom, cur_odom, self.mov_cov_)
                sucess, scan_match_pose =  p._scan_matching(self.init_scan, self.prev_scan, cur_scan, pred_pose)
                predict_elapsed = time.time() - predict_time
                print(f"1 for one predict and match it spends------{predict_elapsed:.5f} seconds")
                if not sucess:
                    # use motion model for pose estimate
                    est_pose = pred_with_noise
                    p.weight_ = p.weight_ * models.measurement_model(cur_scan, pred_with_noise, p.occupied_pts_.T, p.MAP_)          
                else:
                    # sample around scan match pose
                    sample_poses = p._sample_poses_in_interval(scan_match_pose)
                    est_pose = p._compute_new_pose(cur_scan, self.prev_odom, cur_odom, sample_poses)
                    if np.isinf(est_pose).any():
                        print("est_pose invalid. Continue")
                        continue
                self.weights_[i] = p.weight_
                update_time = time.time()  # 记录函数开始时间 
                p._update_map(cur_scan, est_pose)
                update_elapsed = time.time() - update_time
                print(f"2 for update_elapsed it spends------{update_elapsed:.5f} seconds")
            self.Neff = 1 / np.linalg.norm(self.weights_)
            if self.Neff < self.Neff_thresh_:
                self._resample()
            self.prev_scan = cur_scan
            self.prev_odom = cur_odom            
            elapsed_time = time.time() - start_time
            print(f"3 for one loop it spends------{elapsed_time:.5f} seconds-----number of particles = {self.num_p_}-")
        #self.get_logger().info(f"------{elapsed_time:.5f} seconds-----number of particles = {self.num_p_}----time range = {t_end}----")
        # Generate the map based on the partichle with the larges weight.
        combined_log_odds, combined_traj = self._combine_maps(5)
        grid_msg = self._gen_map(self.particles_[np.argmax(self.weights_)], combined_log_odds, combined_traj)
        return grid_msg
        
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


    def _gen_map(self, particle, combined_log_odds, combined_traj):
        '''
            Generates and publishes the combined map with trajectory to the ROS topic
        '''
        logodd_thresh = particle.logodd_thresh_
        MAP = particle.MAP_

        occupancy_grid = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)
        wall_indices = np.where(combined_log_odds > logodd_thresh)
        unexplored_indices = np.where(abs(combined_log_odds) < 1e-1)
        occupancy_grid[list(wall_indices[0]), list(wall_indices[1])] = 100
        occupancy_grid[list(unexplored_indices[0]), list(unexplored_indices[1])] = -1
        if (np.all(combined_traj[0] < occupancy_grid.shape[0]) and 
            np.all(combined_traj[1] < occupancy_grid.shape[1])):
            occupancy_grid[combined_traj[0], combined_traj[1]] = 50
        print(f"wall_indices = {wall_indices}")
        #print(f"unexplored_indices = {unexplored_indices}")
        # Create an OccupancyGrid message
        grid_msg = OccupancyGrid()
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = MAP['res']
        grid_msg.info.width = MAP['sizex']
        grid_msg.info.height = MAP['sizey']
        grid_msg.info.origin.position.x = float(MAP['xmin'])
        grid_msg.info.origin.position.y = float(MAP['ymin'])
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = occupancy_grid.flatten().tolist()
        # Generate the visualization map
        MAP_2_display = 255 * np.ones((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
        MAP_2_display[list(wall_indices[0]), list(wall_indices[1]), :] = [0, 0, 0]
        MAP_2_display[list(unexplored_indices[0]), list(unexplored_indices[1]), :] = [150, 150, 150]
        if (np.all(combined_traj[0] < occupancy_grid.shape[0]) and 
            np.all(combined_traj[1] < occupancy_grid.shape[1])):
            MAP_2_display[combined_traj[0], combined_traj[1]] = [70, 70, 228]
        # Save the visualization map
        resized_image = cv2.resize(MAP_2_display, (600, 600))
        cv2.imshow('SLAM Map Display', resized_image)
        key = cv2.waitKey(1)  # 设置等待时间为1毫秒

        if key == ord('q'):  # 如果按下 'q' 键
            cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
            return None  # 或执行其他退出处理
        return grid_msg

    """
    def _save_map(self, particle, t, p_num):
        MAP = self._gen_map(particle)
        file_name = 'logs/New folder/t_'+ str(t)+'_p_'+str(p_num)+'.png'
        cv2.imwrite(file_name, MAP)
        
    def _disp_map(self, particle, t, p_num = 0):
        MAP = self._gen_map(particle, t)
        plt.imshow(MAP)
        plt.title(str(t)+"_p: "+str(p_num))
        plt.show()

    def _mapping_with_known_poses(self, scan_match_odom, scan_match_flag, t0, t_end = None, interval = 1):
        '''
            Uses noiseless odom data to generate entire map
        '''
        t_end = self.data_.lidar_['num_data'] if t_end is None else t_end + 1
        p = self.particles_[0]
        for t in range(t0, t_end, interval):                         
            odom = scan_match_odom[:,t]
            flag = scan_match_flag[t]
            if not flag:
                odom, _ = p._predict(self.data_, t, self.mov_cov_)        
            p._update_map(self.data_, t, odom)           
            if t%50==0:
                self._disp_map(p, t)
            print(t)                  
        self._save_map(p, t, 0)
    """