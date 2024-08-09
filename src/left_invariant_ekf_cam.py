'''this script is the baseline implementation of IEKF with multi-IMU and camera'''

import os
import argparse

import numpy as np

import scipy
import scipy.io
import scipy.linalg
from scipy.spatial.transform import Rotation

from core.utils import *


class Inertial(object):
    def __init__(self, filename):
        '''construct an IMU object with the input .dat file'''

        print(f"[INFO]: loading IMU data from '{filename}'")
        dataset = scipy.io.loadmat(filename)

        self.C_sv        = dataset["C_sv"]
        self.C_vs = np.linalg.inv(self.C_sv)
        self.r_sv_v      = dataset["r_sv_v"]
        self.accel_s_all = dataset["accel_s_all"]
        self.y_omega_all = dataset["y_omega_all"]
        self.y_accel_all = dataset["y_accel_all"]
        self.w_omega     = dataset["w_omega"].flatten()
        self.w_accel     = dataset["w_accel"].flatten()
        self.n_omega     = dataset["n_omega"].flatten()
        self.n_accel     = dataset["n_accel"].flatten()

        print(f"[INFO]: loaded  IMU data from '{filename}'")


class Multi_Inertial(object):
    def __init__(self, filenames):
        '''create a super class that contains multiple IMUs'''
        self.imu_all = []

        for imu_input in filenames:
            imu = Inertial(imu_input)
            self.imu_all.append(imu)

        self.num_imu = len(self.imu_all)

        # get the centroid of IMUs
        self.avg_r_sv_v = 0.0
        for imu in self.imu_all:
            self.avg_r_sv_v += imu.r_sv_v

        self.avg_r_sv_v = self.avg_r_sv_v / self.num_imu


    def get_transformation(self):
        '''get the transformation from vehicle frame to sensor frame (T_sv)'''

        return Cr2T(np.eye(3), self.avg_r_sv_v)


    def get_avg_noise(self):
        '''get average noise of omega and accel'''

        sum_var_inv_omega = 0.0
        sum_var_inv_accel = 0.0

        for imu in self.imu_all:
            n_omega = imu.C_vs @ imu.n_omega
            n_accel = imu.C_vs @ imu.n_accel
            sum_var_inv_omega += 1 / n_omega**2
            sum_var_inv_accel += 1 / n_accel**2

        avg_n_omega = np.sqrt(1 / sum_var_inv_omega)
        avg_n_accel = np.sqrt(1 / sum_var_inv_accel)

        return avg_n_omega, avg_n_accel


    def get_avg_bias_drift(self):
        '''get average noise of omega and accel'''

        sum_var_inv_omega = 0.0
        sum_var_inv_accel = 0.0

        for imu in self.imu_all:
            w_omega = imu.C_vs @ imu.w_omega
            w_accel = imu.C_vs @ imu.w_accel
            sum_var_inv_omega += 1 / w_omega**2
            sum_var_inv_accel += 1 / w_accel**2

        avg_w_omega = np.sqrt(1 / sum_var_inv_omega)
        avg_w_accel = np.sqrt(1 / sum_var_inv_accel)

        return avg_w_omega, avg_w_accel


    def get_avg_measurement(self, k):
        '''get average measurement (omega, accel)'''

        avg_y_accel = 0.0
        avg_y_omega = 0.0

        for imu in self.imu_all:
            y_omega = imu.C_vs @ imu.y_omega_all[k, :]
            y_accel = imu.C_vs @ imu.y_accel_all[k, :]
            avg_y_omega += y_omega
            avg_y_accel += y_accel

        avg_y_omega = avg_y_omega / self.num_imu
        avg_y_accel = avg_y_accel / self.num_imu

        return avg_y_omega, avg_y_accel


def compute_process_model_jocobian(omega, accel):
    A = np.zeros([15, 15])
    omega_vee = get_lifted_form(omega)
    accel_vee = get_lifted_form(accel)

    A[0:3, 0:3] = -omega_vee
    A[3:6, 3:6] = -omega_vee
    A[6:9, 6:9] = -omega_vee
    A[3:6, 0:3] = -accel_vee
    A[6:9, 3:6] = np.eye(3)

    A[0:6, 9:15] = -np.eye(6)
    return A


def compute_measurement_model_jocobian(T, fx, fy, cx, cy, x, y, z, u, v):
    e_y = np.zeros(2)
    p_i = np.array([x, y, z, 1.0])
    p_c = np.matmul(T, p_i) # landmarks in camera frame

    y_c  = np.array([u, v])
    y_op = np.zeros(2)
    y_op[0] = fx *  p_c[0] / p_c[2] + cx # u
    y_op[1] = fy *  p_c[1] / p_c[2] + cy # v
    e_y = y_c - y_op

    Z_jk = -get_circle_dot_for_SE2_3(p_c)

    S_jk = np.zeros([2, 3])
    S_jk[0, 0] =  fx /  p_c[2]
    S_jk[0, 2] = -fx *  p_c[0] / p_c[2]**2
    S_jk[1, 1] =  fy /  p_c[2]
    S_jk[1, 2] = -fy *  p_c[1] / p_c[2]**2
    # print(f"S_jk = {S_jk}")

    D_T = np.zeros([3, 4])
    D_T[0: 3, 0: 3] = np.eye(3)
    G = S_jk @ D_T @ Z_jk

    return e_y, G


def plot_bias(b_omega_all, b_omega_true_all, b_accel_all, b_accel_true_all):
    K = b_omega_all.shape[0]

    fig, axes = plt.subplots(6, 1)
    t = np.linspace(start=0, stop=K, num=K, dtype=int)

    axes[0].plot(t, b_omega_true_all[:, 0])
    axes[1].plot(t, b_omega_true_all[:, 1])
    axes[2].plot(t, b_omega_true_all[:, 2])

    axes[0].plot(t, b_omega_all[:, 0])
    axes[1].plot(t, b_omega_all[:, 1])
    axes[2].plot(t, b_omega_all[:, 2])

    axes[3].plot(t, b_accel_true_all[:, 0])
    axes[4].plot(t, b_accel_true_all[:, 1])
    axes[5].plot(t, b_accel_true_all[:, 2])

    axes[3].plot(t, b_accel_all[:, 0])
    axes[4].plot(t, b_accel_all[:, 1])
    axes[5].plot(t, b_accel_all[:, 2])

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", help="Path to .mat file with the simulated trajectory data", default="./res/dataset/data_trajectory.mat",  required=False)
    parser.add_argument("--cam",        help="Path to .mat file with the simulated camera data",     default="./res/dataset/data_cam.mat",         required=False)
    parser.add_argument("--imu",        help="Path to .mat file with the simulated imu data",        default=[], action="append",                  required=False)
    args = parser.parse_args()

    #############
    # load data #
    #############
    # load trajectory data
    dataset = scipy.io.loadmat(args.trajectory)
    print("[INFO]: loading trajectory data...")

    N = int(dataset["N"][0][0])
    dt = dataset["dt"][0][0]
    C_iv_all    = dataset["C_iv_all"]
    omega_v_all = dataset["omega_v_all"]
    alpha_v_all = dataset["alpha_v_all"]
    r_vi_i_all  = dataset["r_vi_i_all"]
    vel_i_all   = dataset["vel_i_all"]
    accel_v_all = dataset["accel_v_all"]
    accel_i_all = dataset["accel_i_all"]

    print("[INFO]: loaded trajectory data")

    # load IMU data
    print("[INFO]: loading IMU data...")
    virtual_imu = Multi_Inertial(args.imu)
    print("[INFO]: loaded IMU data")

    # load cam data
    dataset = scipy.io.loadmat(args.cam)
    print("[INFO]: loading camera data...")

    C_vc        = dataset["C_vc"]
    C_cv        = np.linalg.inv(C_vc)
    r_cv_v      = dataset["r_cv_v"]
    fx          = float(dataset["fx"][0][0])
    fy          = float(dataset["fy"][0][0])
    cx          = dataset["cx"][0][0]
    cy          = dataset["cy"][0][0]
    var_n_u     = dataset["var_n_u"]
    var_n_v     = dataset["var_n_v"]
    obs_all     = dataset["obs_all"]
    M           = int(dataset["M"][0][0])

    print("[INFO]: loaded camera data")

    # compute quantity
    T_imu = 10    # period of IMU: sample every T_imu timestep
    T_cam = 500   # period of cam: sample every T_cam timestep
    dt = dt * T_imu

    (n_omega, n_accel) = virtual_imu.get_avg_noise()
    Q_n_omega = np.diag(n_omega**2 / dt) # n_omega is in unit of (rad/s) / sqrt(Hz) thus var = n_omega**2 / dt
    Q_n_accel = np.diag(n_accel**2 / dt) # n_accel is in unit of (m/s^2) / sqrt(Hz) thus var = n_accel**2 / dt
    Q_n = scipy.linalg.block_diag(Q_n_omega, Q_n_accel, np.zeros([3, 3]))

    (w_omega, w_accel) = virtual_imu.get_avg_bias_drift()
    Q_b_omega = np.diag(w_omega**2 / dt) # w_omega is in unit of (rad/s) * sqrt(Hz) thus var = w_omega**2 / dt
    Q_b_accel = np.diag(w_accel**2 / dt) # w_accel is in unit of (m/s^2) * sqrt(Hz) thus var = w_accel**2 / dt
    Q_b = scipy.linalg.block_diag(Q_b_omega, Q_b_accel)

    R_cam = np.eye(2)
    R_cam[0, 0] = var_n_u[0][0]
    R_cam[1, 1] = var_n_v[0][0]
    R_cam_all = [R_cam] * M
    R = scipy.linalg.block_diag(*R_cam_all)

    # relative transform from vehicle to virtual IMU
    T_sv = virtual_imu.get_transformation()

    ############
    # run IEKF #
    ############
    # initialize
    C_iv = C_iv_all[:, :, 0]
    vel = vel_i_all[0, :]
    pos = r_vi_i_all[0, :]
    xi = np.zeros(9) # state in term of error
    b_omega = np.zeros(3) #TODO::initialize with initial value
    b_accel = np.zeros(3)
    # b_omega = np.array([0.1491, 0.1234, 0.1028])
    # b_accel = np.array([0.4905, 0.3454, 0.2392])
    P = np.eye(15) * 1e-6

    C_est_all   = np.zeros_like(C_iv_all)
    vel_est_all = np.zeros_like(vel_i_all)
    pos_est_all = np.zeros_like(r_vi_i_all)
    xi_all = np.zeros([N, 9])
    e_b_all = np.zeros([N, 6])
    b_omega_all = np.zeros([N, 3])
    b_accel_all = np.zeros([N, 3])
    b_omega_true_all = np.zeros([N, 3])
    b_accel_true_all = np.zeros([N, 3])
    P_all  = np.zeros([15, 15, N])

    C_est_all[:, :, 0] = np.copy(C_iv)
    vel_est_all[0, :]  = np.copy(vel)
    pos_est_all[0, :]  = np.copy(pos)
    P_all[:, :, 0]     = np.copy(P)

    # N = 50000
    for k in range(N):
        # get imu measurement
        y_omega, y_accel = virtual_imu.get_avg_measurement(k)

        # run propagation
        if k % T_imu == 0:
            # get measurement
            omega = y_omega - b_omega
            accel = y_accel - b_accel
            accel_i = np.matmul(C_iv, accel) - np.array([0.0, 0.0, 9.81])

            # propagate rotation
            C = Rotation.from_rotvec(omega * dt).as_matrix()
            C_iv = C_iv @ C
            C_vi = np.linalg.inv(C_iv)

            # propagate velocity and position
            pos = pos + 0.5 * accel_i * dt**2 + vel * dt
            vel = vel + 1.0 * accel_i * dt

            # propagate covariance
            A = compute_process_model_jocobian(omega, accel)
            F = scipy.linalg.expm(A * dt)
            P = F @ P @ np.transpose(F) + scipy.linalg.block_diag(Q_n, Q_b) * dt**2

        # run correction
        if k % T_cam == (T_cam - 1):
            T_vi = Cr2T(C_vi, pos)
            T_cv = Cr2T(C_cv, r_cv_v)
            T_ci = T_cv @ T_vi

            z_k = np.zeros(2 * M)
            G_k = np.zeros([2 * M, 15])

            for m in range(M):
                [x, y, z, u, v, n_u, n_v] = obs_all[k, m, :].flatten().tolist()
                u += n_u
                v += n_v
                z_km, G_km = compute_measurement_model_jocobian(T_ci, fx, fy, cx, cy, x, y, z, u, v)
                z_k[m * 2 : (m + 1) * 2] = z_km
                G_k[m * 2 : (m + 1) * 2, 0: 9] = G_km
                G_k_T = np.transpose(G_k)

            # compute Kalman gain
            S_k = G_k @ P @ G_k_T + R
            K_k = P @ G_k_T @ np.linalg.inv(S_k)

            # update covariance
            P = (np.eye(15) - K_k @ G_k) @ P

            # update state
            dX = K_k @ z_k
            vel += C_iv @ dX[3: 6]
            pos += C_iv @ dX[6: 9]
            dC = Rotation.from_rotvec(dX[0: 3]).as_matrix()
            C_iv = C_iv @ dC

            # update bias
            b_omega += dX[9 : 12]
            b_accel += dX[12: 15]

        # save intermmediate result
        C_est_all[:, :, k] = np.copy(C_iv)
        vel_est_all[k, :]  = np.copy(vel)
        pos_est_all[k, :]  = np.copy(pos)
        P_all[:, :, k]     = np.copy(P)

        b_omega_all[k, :] = b_omega
        b_accel_all[k, :] = b_accel

        # get true bias
        b_omega_true = y_omega - omega_v_all[k, :]
        b_accel_true = (y_accel - np.matmul(np.linalg.inv(C_iv_all[:, :, k]), np.array([0.0, 0.0, 9.81]))) - accel_v_all[k, :]
        b_omega_true_all[k, :] = b_omega_true
        b_accel_true_all[k, :] = b_accel_true

        # compute bias error
        e_b_all[k, 0: 3] = b_omega_true - b_omega
        e_b_all[k, 3: 6] = b_accel_true - b_accel

        # compute state error
        C_err    = np.linalg.inv(C_iv) @ C_iv_all[:, :, k]
        xi[0: 3] = Rotation.from_matrix(C_err).as_rotvec()
        xi[3: 6] = vel -  vel_i_all[k, :]
        xi[6: 9] = pos - r_vi_i_all[k, :]
        xi_all[k, :] = np.copy(xi)

    print("[INFO]: done estimating the trajectory")

    ################
    # plot results #
    ################
    # plot trajectory
    plot_trajectory(r_vi_i_all[0: N, :], pos_est_all[0: N, :])

    # plot state error
    plot_state_error_and_uncertainty(xi_all[0: N, 0: 3], P_all[0: 3, 0: 3, 0: N], titles=["e_roll", "e_pitch", "e_yaw"])
    plot_state_error_and_uncertainty(xi_all[0: N, 3: 6], P_all[3: 6, 3: 6, 0: N], titles=["e_vel_x", "e_vel_y", "e_vel_z"])
    plot_state_error_and_uncertainty(xi_all[0: N, 6: 9], P_all[6: 9, 6: 9, 0: N], titles=["e_pos_x", "e_pos_y", "e_pos_z"])

    # plot bias error
    plot_state_error_and_uncertainty(e_b_all[0: N, 0: 3], P_all[ 9: 12,  9: 12, 0: N], titles=["b_omega_x", "b_omega_y", "b_omega_z"])
    plot_state_error_and_uncertainty(e_b_all[0: N, 3: 6], P_all[12: 15, 12: 15, 0: N], titles=["b_accel_x", "b_accel_y", "b_accel_z"])

    # plot bias
    plot_bias(b_omega_all, b_omega_true_all, b_accel_all, b_accel_true_all)

    print("[INFO]: done")


if __name__ == "__main__":
    main()