import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

from core.sensor import *
from core.inertial_navigation_system import *


class VIMU_estimator(inertial_navigation_system):
    '''estimator with single-IMU or virtual-IMU using vanilla IEKF'''

    def __init__(self, T_iv):
        super().__init__()
        T_vi = np.linalg.inv(T_iv)
        self.C_iv, self.r_vi_i = pose_to_rotation_and_translation(T_iv)
        self.vel  = np.zeros(3)
        self.b_omega = np.zeros(3)
        self.b_accel = np.zeros(3)
        self.covariance = np.eye(15) * 1e-6


    def get_state_estimate(self):
        return self.C_iv, self.vel, self.r_vi_i


    def get_state_covariance(self):
        return self.covariance


    def handle_IMU_measurement(self, id, dt, omega, accel, R=None):
        '''
        update state and covariance with IMU measurement
            - id: string id of the IMU sensor
            - dt: time elapsed since previous update
            - omega: angular velocity measurement
            - accel: linear acceleration measurement
            - R: noise covariance matrix
        '''

        # get measurement
        omega = omega - self.b_omega
        accel = accel - self.b_accel
        accel_i = np.matmul(self.C_iv, accel) - np.array([0.0, 0.0, 9.81])

        # propagate rotation
        C = Rotation.from_rotvec(omega * dt).as_matrix()
        self.C_iv = self.C_iv @ C

        # propagate velocity and position
        self.r_vi_i = self.r_vi_i + 0.5 * accel_i * dt**2 + self.vel * dt
        self.vel = self.vel + 1.0 * accel_i * dt

        # compute process model Jacobian
        A = compute_process_model_jocobian(omega, accel)
        F = scipy.linalg.expm(A * dt)

        # construct noise covariance
        if R == None:
            Q_n_omega = np.diag(self.imu[id].n_omega**2)
            Q_n_accel = np.diag(self.imu[id].n_accel**2)
            Q_n = scipy.linalg.block_diag(Q_n_omega, Q_n_accel, np.zeros([3, 3]))

            Q_b_omega = np.diag(self.imu[id].w_omega**2)
            Q_b_accel = np.diag(self.imu[id].w_accel**2)
            Q_b = scipy.linalg.block_diag(Q_b_omega, Q_b_accel)

            R = scipy.linalg.block_diag(Q_n, Q_b)

        # propagate state covariance
        self.covariance = F @ self.covariance @ np.transpose(F) + R * dt

        return self.C_vi, self.vel, self.r_vi_i, self.covariance


    def handle_camera_measurement(self, id, pt_3d, kp_2d, R=None):
        '''
        update state and covariance with camera measurement
            - id: string id of the camera
            - pt_3d: landmark in 3D
            - kp_2d: keypoint observation in image plane
            - R: noise covariance matrix
        '''

        # get transformations
        T_iv = rotation_and_translation_to_pose(self.C_iv, self.r_vi_i)
        T_vi = np.linalg.inv(T_iv)
        T_cv = self.camera[id].T_cv
        T_ci = T_cv @ T_vi

        # get camera properties
        K = self.camera[id].K
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # construct measurement Jacobian
        M = kp_2d.shape[0]
        z_k = np.zeros(2 * M)
        G_k = np.zeros([2 * M, 15])

        for m in range(M):
            x = pt_3d[m, 0]
            y = pt_3d[m, 1]
            z = pt_3d[m, 2]
            u = kp_2d[m, 0]
            v = kp_2d[m, 1]
            z_km, G_km = compute_measurement_model_jocobian(T_ci, fx, fy, cx, cy, x, y, z, u, v)
            z_k[m * 2 : (m + 1) * 2] = z_km
            G_k[m * 2 : (m + 1) * 2, 0: 9] = G_km
            G_k_T = np.transpose(G_k)

        # construct noise covariance
        R_cam = np.eye(2)
        R_cam[0, 0] = self.camera[id].var_n_u
        R_cam[1, 1] = self.camera[id].var_n_v
        R_cam_all = [R_cam] * M
        R = scipy.linalg.block_diag(*R_cam_all)

        # compute Kalman gain
        S_k = G_k @ self.covariance @ G_k_T + R
        K_k = self.covariance @ G_k_T @ np.linalg.inv(S_k)

        # update covariance
        self.covariance = (np.eye(15) - K_k @ G_k) @ self.covariance

        # update state
        dX = K_k @ z_k
        self.vel    += self.C_iv @ dX[3: 6]
        self.r_vi_i += self.C_iv @ dX[6: 9]
        dC = Rotation.from_rotvec(dX[0: 3]).as_matrix()
        self.C_iv = self.C_iv @ dC

        # update bias
        self.b_omega += dX[ 9: 12]
        self.b_accel += dX[12: 15]

        return self.C_iv, self.vel, self.r_vi_i, self.covariance


    def handle_GPS_measurement(self, id, y, R=None):
        '''
        update state and covariance with GPS measurement
            - id: string id of the GPS sensor
            - y: GPS position measurement
            - R: noise covariance matrix
        '''

        # compute error (innovation)
        z_k = y - self.r_vi_i

        # construct Jacobian
        G_k = np.zeros([3, 9])
        G_k[0: 3, 6: 9] = np.eye(3)
        G_k_T = np.transpose(G_k)

        # compute Kalman gain
        S_k = G_k @ self.covariance @ G_k_T + R
        K_k = self.covariance @ G_k_T @ np.linalg.inv(S_k)

        # update covariance
        self.covariance = (np.eye(9) - K_k @ G_k) @ self.covariance

        # update state
        dX = K_k @ z_k
        self.vel    += dX[3: 6]
        self.r_vi_i += dX[6: 9]
        dC = Rotation.from_rotvec(dX[0: 3]).as_matrix()
        self.C_iv = self.C_iv @ dC

        return self.C_iv, self.vel, self.r_vi_i, self.covariance