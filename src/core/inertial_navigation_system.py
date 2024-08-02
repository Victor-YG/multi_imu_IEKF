import numpy as np

from core.sensor import *


class inertial_navigation_system(object):
    '''Interface for IMU-based state estimator'''

    def __init__(self):
        self.C_vi = None
        self.r_vi_i = None
        self.covariance = None

        self.imu = dict()
        self.imu_model = dict()
        self.camera = dict()
        self.gps = dict()


    def get_state_estimate(self):
        return None


    def get_state_covariance(self):
        return None


    def add_IMU(self, id, T_sv, n_omega, n_accel, w_omega, w_accel):
        '''
        add IMU sensor as input to the INS
            - id: string id of the IMU sensor
            - T_sv: sensor pose relative to vehicle
            - n_omega: noise of omega
            - n_accel: noise of accel
            - w_omega: bias drift rate of omega
            - w_accel: bias drift rate of accel
        '''

        self.imu[id] = IMU_sensor_prop(id, T_sv, n_omega, n_accel, w_omega, w_accel)
        # self.imu_model[id] = IMU_sensor_model_kalman()
        self.imu_model[id] = IMU_sensor_model_lowpass()


    def add_IMU(self, id, imu_sensor_prop):
        self.imu[id] = imu_sensor_prop
        # self.imu_model[id] = IMU_sensor_model_kalman()
        self.imu_model[id] = IMU_sensor_model_lowpass()


    def add_camera(self, id, T_cv, K, var_n_u, var_n_v):
        '''
        add camera as input to the INS
            - id: string id of the camera
            - T_cv: camera pose relative to vehicle
            - K: intrinsic matrix of the camera
            - var_n_u: noise variance in u direction
            - var_n_v: noise variance in v direction
        '''

        self.camera[id] = camera_sensor_prop(id, T_cv, K, var_n_u, var_n_v)


    def add_camera(self, id, cam_sensor_prop):
        self.camera[id] = cam_sensor_prop


    def add_GPS(self, id, T_sv, R):
        '''
        add GPS sensor as input to the INS
            - id: string id of the GPS sensor
            - T_sv: sensor pose relative to vehicle
            - R: noise covariance matrix
        '''

        self.gps[id] = GPS_sensor_prop(id, T_sv, R)


    def add_GPS(self, id, gps_sensor_prop):
        self.gps[id] = gps_sensor_prop


    def handle_IMU_measurement(self, id, dt, omega, accel, R=None):
        '''
        update state and covariance with IMU measurement
            - id: string id of the IMU sensor
            - dt: time elapsed since previous update
            - omega: angular velocity measurement
            - accel: linear acceleration measurement
            - R: noise covariance matrix
        '''
        return self.C_vi, self.r_vi_i, self.covariance


    def handle_camera_measurement(self, id, pt_3d, kp_2d, R=None):
        '''
        update state and covariance with camera measurement
            - id: string id of the camera
            - pt_3d: landmark in 3D
            - kp_2d: keypoint observation in image plane
            - R: noise covariance matrix
        '''
        return self.C_vi, self.r_vi_i, self.covariance


    def handle_GPS_measurement(self, id, y, R=None):
        '''
        update state and covariance with GPS measurement
            - id: string id of the GPS sensor
            - y: GPS position measurement
            - R: noise covariance matrix
        '''
        return self.C_vi, self.r_vi_i, self.covariance