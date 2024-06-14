import os
import argparse

import scipy.io
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to .mat file with the simulation data", default="./res/dataset/data_imu_offset_10_with_bias.mat", required=False)
    args = parser.parse_args()

    # load data
    dataset = scipy.io.loadmat(args.dataset)
    print("[INFO]: loaded simulation data")

    N = int(dataset["N"][0])
    dt = dataset["dt"]
    C_all   = dataset["C_all"]
    omega_all = dataset["omega_all"]
    alpha_all = dataset["alpha_all"]
    pos_all = dataset["pos_all"]
    vel_all = dataset["vel_all"]
    accel_all = dataset["accel_all"]
    accel_s_all = dataset["accel_s_all"]
    accel_i_all = dataset["accel_i_all"]
    y_omega_all = dataset["y_omega_all"]
    y_accel_all = dataset["y_accel_all"]
    C_sv   = dataset["C_sv"]
    r_sv_v = dataset["r_sv_v"]
    C_vs = np.linalg.inv(C_sv)
    print("[INFO]: parsed simulation data")

    # solve trajectories by integration
    pos_est_all = np.zeros_like(pos_all)
    pos_off_all = np.zeros_like(pos_all)
    C_est_all   = np.zeros_like(C_all)

    C_iv = C_all[:, :, 0]
    pos  = pos_all[0, :] + np.matmul(C_iv, r_sv_v.reshape(3, 1)).flatten()
    vel  = vel_all[0, :] + np.matmul(C_iv, np.cross(omega_all[0, :], r_sv_v).reshape(3, 1)).flatten()
    pos_off = np.zeros_like(pos)

    for k in range(N):
        pos_off = pos_all[k, :] + np.matmul(C_all[:, :, k], r_sv_v.reshape(3, 1)).flatten()
        pos_off_all[k, :] = pos_off
        pos_est_all[k, :] = pos
        C_est_all[:, :, k] = C_iv

        omega = omega_all[k, :]
        accel = y_accel_all[k, :]
        C_is    = np.matmul(C_iv, C_vs)

        # selecting how to populate acceleration in inertial frame
        accel_i = np.matmul(C_iv, accel) - np.array([0.0, 0.0, 9.81])           # use measurement
        # accel_i = np.matmul(C_is, accel_s_all[k, :].reshape(3, 1)).flatten()    # use GT acceleration in sensor frame

        # integrate
        pos = pos + 0.5 * accel_i * dt**2 + vel * dt
        vel = vel + 1.0 * accel_i * dt

        # update rotation
        R = Rotation.from_rotvec(omega * dt).as_matrix()
        C_iv = np.matmul(C_iv, R)

    # plot trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_all[:, 0], pos_all[:, 1], pos_all[:, 2], c='g', label='vehicle_trajectory_gt')
    ax.plot(pos_off_all[:, 0], pos_off_all[:, 1], pos_off_all[:, 2], c='b', label='sensor_trajectory_gt')
    ax.plot(pos_est_all[:, 0], pos_est_all[:, 1], pos_est_all[:, 2], c='r', label='sensor_Trajectory_est')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()
    print("[INFO]: done estimating the trajectory")


if __name__ == "__main__":
    main()