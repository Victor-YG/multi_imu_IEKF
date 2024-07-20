'''this script is the baseline implementation of IEKF with IMU and GPS'''

import os
import argparse

import numpy as np

import scipy
import scipy.io
import scipy.linalg
from scipy.spatial.transform import Rotation

from core.utils import *


def compute_process_model_jocobian(omega, accel):
    A = np.zeros([9, 9])
    omega_vee = get_lifted_form(omega)
    accel_vee = get_lifted_form(accel)
    A[0:3, 0:3] = -omega_vee
    A[3:6, 3:6] = -omega_vee
    A[6:9, 6:9] = -omega_vee
    A[3:6, 0:3] = -accel_vee
    A[6:9, 3:6] = np.eye(3)
    return A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", help="Path to .mat file with the simulated trajectory data", default="./res/dataset/data_trajectory.mat", required=False)
    parser.add_argument("--imu",        help="Path to .mat file with the simulated imu data",        default="./res/dataset/data_imu.mat",        required=False)
    parser.add_argument("--gps",        help="Path to .mat file with the simulated gps data",        default="./res/dataset/data_gps.mat",        required=False)
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
    dataset = scipy.io.loadmat(args.imu)
    print("[INFO]: loading IMU data...")

    C_sv        = dataset["C_sv"]
    r_sv_v      = dataset["r_sv_v"]
    accel_s_all = dataset["accel_s_all"]
    y_omega_all = dataset["y_omega_all"]
    y_accel_all = dataset["y_accel_all"]
    w_omega     = dataset["w_omega"]
    w_accel     = dataset["w_accel"]
    n_omega     = dataset["n_omega"].flatten()
    n_accel     = dataset["n_accel"].flatten()

    C_vs = np.linalg.inv(C_sv)
    print("[INFO]: loaded IMU data")

    # load GPS data
    dataset = scipy.io.loadmat(args.gps)
    print("[INFO]: loading GPS data...")

    var_n_x     = dataset["var_n_x"][0][0]
    var_n_y     = dataset["var_n_y"][0][0]
    var_n_z     = dataset["var_n_z"][0][0]
    pos_obs_all = dataset["pos_obs_all"]

    print("[INFO]: loaded GPS data")

    # compute quantity
    T_imu = 10   # period of IMU: sample every T_imu timestep
    T_gps = 1000 # period of GPS: sample every T_gps timestep
    dt = dt * T_imu

    Q_omega = np.diag(n_omega**2 / dt) # n_omega is in unit of (rad/s) / sqrt(Hz) thus var = n_omega**2 / dt
    Q_accel = np.diag(n_accel**2 / dt) # n_accel is in unit of (  m/s) / sqrt(Hz) thus var = n_accel**2 / dt
    Q = scipy.linalg.block_diag(Q_omega, Q_accel, np.zeros([3, 3]))

    R = np.eye(3)
    R[0, 0] = var_n_x
    R[1, 1] = var_n_y
    R[2, 2] = var_n_z

    ############
    # run IEKF #
    ############
    # initialize
    C_iv = C_iv_all[:, :, 0]
    vel = vel_i_all[0, :]
    pos = r_vi_i_all[0, :]
    xi = np.zeros(9) # state in term of error
    P  = np.eye(9) * 1e-6

    C_est_all   = np.zeros_like(C_iv_all)
    vel_est_all = np.zeros_like(vel_i_all)
    pos_est_all = np.zeros_like(r_vi_i_all)
    xi_all = np.zeros([N, 9])
    P_all  = np.zeros([9, 9, N])

    C_est_all[:, :, 0] = np.copy(C_iv)
    vel_est_all[0, :]  = np.copy(vel)
    pos_est_all[0, :]  = np.copy(pos)
    P_all[:, :, 0]     = np.copy(P)

    # N = 4000
    for k in range(N):
        # run propagation
        if k % T_imu == 0:
            # get data
            omega = y_omega_all[k, :]
            accel = y_accel_all[k, :]
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
            P = F @ P @ np.transpose(F) + Q

            # TODO::implement propagation for bias

        # run correction
        if k % T_gps == 0:
            # compute error (innovation)
            z_k = pos_obs_all[k] - pos

            # construct Jacobian
            G_k = np.zeros([3, 9])
            G_k[0: 3, 6: 9] = np.eye(3)
            G_k_T = np.transpose(G_k)

            # compute Kalman gain
            S_k = G_k @ P @ G_k_T + R
            K_k = P @ G_k_T @ np.linalg.inv(S_k)

            # update covariance
            P = (np.eye(9) - K_k @ G_k) @ P

            # update state
            dX = K_k @ z_k
            vel += dX[3: 6]
            pos += dX[6: 9]
            dC = Rotation.from_rotvec(dX[0: 3]).as_matrix()
            C_iv = C_iv @ dC

        # save intermmediate result
        C_est_all[:, :, k] = np.copy(C_iv)
        vel_est_all[k, :]  = np.copy(vel)
        pos_est_all[k, :]  = np.copy(pos)
        P_all[:, :, k]     = np.copy(P)

        # compute error
        C_err   = np.linalg.inv(C_iv_all[:, :, k]) @ C_iv
        xi[0: 3] = Rotation.from_matrix(C_err).as_rotvec()
        xi[3: 6] = vel -  vel_i_all[k, :]
        xi[6: 9] = pos - r_vi_i_all[k, :]
        xi_all[k, :] = np.copy(xi)

    print("[INFO]: done estimating the trajectory")

    ################
    # plot results #
    ################
    # plot trajectory
    plot_trajectory(r_vi_i_all[0: N], pos_est_all[0: N])

    # plot error
    plot_error_distribution(xi_all[0: N], P_all[0: N])

    print("[INFO]: done")


if __name__ == "__main__":
    main()