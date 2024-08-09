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
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
import time
from geometry_msgs.msg import Pose, Point

def_zero_threshold = 1e-15


class SLAM():
    def __init__(self, mov_cov, num_p = 10, map_resolution = 0.05, map_dimension = 10, Neff_thresh = 3):
        self.num_p_ = num_p
        self.Neff_ = 0
        self.Neff_thresh_ = Neff_thresh
        self.weights_ = np.ones(num_p) / num_p
        self.mov_cov_ = mov_cov

        self.lidar_data = None
        self.odom_data_list = []
        self.particles_ = []
        self.Neff_history = []
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
        #print(f"_resample --EXIT--- self.Neff ={self.Neff}")

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
        cur_scan = self.lidar_data
        cur_odom = self.get_odom_at_time(cur_scan)
        for i, p in enumerate(self.particles_):
            # predict with motion model
            pred_pose, pred_with_noise = p._predict(self.prev_odom, cur_odom, self.mov_cov_)
            #is_matched, scan_match_pose =  p._scan_matching(self.init_scan, self.prev_scan, cur_scan, pred_pose)
            is_matched = False
            if not is_matched:
                if p.occupied_pts_.T.size == 0:
                    continue
                # use motion model for pose estimate
                est_pose = pred_with_noise
                measure = models.measurement_model(cur_scan, pred_with_noise, p.occupied_pts_.T, p.MAP_)
                if measure > def_zero_threshold :
                    # the weight update is based on the measurement likelihood from measurement model.
                    p.weight_ = p.weight_ * measure
                else :
                    continue
            else:
                print(f"is matched.  going to calculate sample poses")
                #sample_poses = p._sample_poses_in_interval(scan_match_pose)
                #est_pose = p._compute_new_pose(cur_scan, self.prev_odom, cur_odom, sample_poses)

            self.weights_[i] = p.weight_
            p._update_map(cur_scan, est_pose)

        self.Neff = 1 / np.sum(self.weights_ ** 2)
        if not np.isfinite(self.Neff):
                self.Neff = 0
        self.Neff_history.append(self.Neff)
        plt.figure(figsize=(10, 6))
        plt.plot(self.Neff_history)
        plt.xlabel('Time (iterations)')
        plt.ylabel('Neff')
        plt.title('Effective Sample Size (Neff) Over Time')
        plt.savefig('/home/agilex/slam_logs/neff_plot.png')
        plt.close()
        if self.Neff < self.Neff_thresh_:
            self._resample()
        self.prev_scan = cur_scan
        self.prev_odom = cur_odom
        # Generate the map based on the partichle with the larges weight.
        return self.particles_[np.argmax(self.weights_)]

    def _save_map(self, particle, t, p_num):
        MAP = self._gen_map(particle)
        file_name = 'logs/New folder/t_'+ str(t)+'_p_'+str(p_num)+'.png'
        cv2.imwrite(file_name, MAP)
