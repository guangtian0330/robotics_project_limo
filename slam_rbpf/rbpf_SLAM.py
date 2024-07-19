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
from geometry_msgs.msg import Pose, Point

def_zero_threshold = 1e-15


class SLAM():
    def __init__(self, mov_cov, num_p = 10, map_resolution = 0.05, map_dimension = 30, Neff_thresh = 3):
        self.num_p_ = num_p
        self.Neff_ = 0
        self.Neff_thresh_ = Neff_thresh
        self.weights_ = np.ones(num_p) / num_p
        self.mov_cov_ = mov_cov
        self.set_key_scan = False
        self.key_scan = None
        self.lidar_data = None
        self.odom_data_list = []
        self.particles_ = []
        self.grid_msg = None
        self.map_img = None
        self.map_msg = None
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
        print(f"__init_map_for_particles")
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
            is_matched = False
            # It's initially supposed to set a key scan and pose for each particle.
            if self.set_key_scan:
                self.key_scan = cur_scan
                print(f"---set key data for particle{i}-----------------")
                p._set_key_data(pred_with_noise)
            elif self.key_scan is not None and np.linalg.norm(np.array(cur_odom[:2]) - np.array(self.prev_odom[:2])) >= 0.01:
                # Use the key scan and key pose to do the match for loop_closure detection.
                # It has to be sure that the limo should be moving.
                print(f"---scan_matching started-----------------")
                #is_matched, scan_match_pose =  p._scan_matching(self.init_scan, self.key_scan, cur_scan, pred_pose)
            #print(f"|----_run_slam pred_pose = {pred_pose}, pred_with_noise = {pred_with_noise}, scan_match_pose = {scan_match_pose}, sucess={sucess}")
            if is_matched: 
                # if it's detected matched with the key scan, create a list of samples and calculate the improved poses.
                #sample_poses = p._sample_poses_in_interval(scan_match_pose)
                #est_pose = p._compute_new_pose(cur_scan, self.prev_odom, cur_odom, sample_poses)
                is_matched = np.isfinite(est_pose).all()
            if not is_matched:
                if p.occupied_pts_.T.size == 0:
                    print(f"---p.occupied_pts_.T.size = {p.occupied_pts_.T.size}-----------------")
                    continue
                # use motion model for pose estimate
                est_pose = pred_with_noise
                measure = models.measurement_model(cur_scan, pred_with_noise, p.occupied_pts_.T, p.MAP_)
                print(f"|-----measurement_model, current weight{i} is {self.weights_[i]}, measure = {measure}----------------")
                if measure > def_zero_threshold :
                    p.weight_ = p.weight_ * measure
                else :
                    print(f"|-----no change for particle{i}, and current weight is {self.weights_[i]}, measure = {measure}-----")
                    continue
            else: # if the current pose mactches the key pose and scan, then update the key scan and pose.
                self.key_scan = cur_scan
                p._set_key_data(pred_with_noise)
            self.weights_[i] = p.weight_
            #update_time = time.time()
            p._update_map(cur_scan, est_pose)
            #update_elapsed = time.time() - update_time
            print(f"|-----_update_map  finished ------ self.weights_[{i}] = {self.weights_[i]}")
            print(f"----------------------------------------------------------------------------------------")
        if self.set_key_scan:
            self.set_key_scan = False
        self.Neff = 1 / np.sum(self.weights_ ** 2)
        print(f"Calculating the neff value after running all particles.------ self.Neff = {self.Neff}")
        if self.Neff < self.Neff_thresh_:
            self._resample()
        self.prev_scan = cur_scan
        self.prev_odom = cur_odom
        #print(f"---_run_slam---{(slam_time - run_start_time):.5f} seconds-----number of particles = {self.num_p_}-----")
        # Generate the map based on the partichle with the larges weight.
        print(f"----The updated pose is {self.particles_[np.argmax(self.weights_)].traj_indices_[:,-1]} from particle[{np.argmax(self.weights_)}]---------")
        return self.particles_[np.argmax(self.weights_)]

    """
    def _save_map(self, particle, t, p_num):
        MAP = self._gen_map(particle)
        file_name = 'logs/New folder/t_'+ str(t)+'_p_'+str(p_num)+'.png'
        cv2.imwrite(file_name, MAP)
    """