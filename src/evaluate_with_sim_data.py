'''This script evaluates different multi-IMU algorithm using simulated data'''

import os
import argparse

import scipy.io

from core.VIMU_estimator import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", help="Path to .mat file with the simulated trajectory data", default=None, required=True)
    parser.add_argument("--imu",        help="Path to .mat file with the simulated imu data",        default=[], action="append", required=True)
    parser.add_argument("--cam",        help="Path to .mat file with the simulated camera data",     default=[], action="append", required=False)
    parser.add_argument("--gps",        help="Path to .mat file with the simulated gps data",        default=[], action="append", required=False)
    parser.add_argument("--algo",       help="State estimation algorithm to evaluate (VIMU or CM-IMU)", default="VIMU")
    args = parser.parse_args()

    estimator = None
    T_imu = 10
    T_cam = 500
    T_gps = 1000

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
    imu_sensors = []
    print("[INFO]: loading IMU data...")
    if args.algo == "VIMU":
        virtual_imu = VIMU_sensor("vimu", args.imu)
        virtual_imu.set_period(T_imu)
        estimator = VIMU_estimator(virtual_imu.get_transformation())
        estimator.add_IMU(virtual_imu.id, virtual_imu.get_properties())
        imu_sensors.append(virtual_imu)
    elif args.algo == "SIMU":
        imu = IMU_sensor(f"imu-0", args.imu[0])
        imu.set_period(T_imu)
        estimator = VIMU_estimator(imu.get_transformation())
        estimator.add_IMU(imu.id, imu.get_properties())
        imu_sensors.append(imu)
    elif args.algo == "CIMU":
        pass
        #TODO::to be implemented
    else:
        exit(f"[FAIL]: unsupported estimator algorithm '{args.algo}'")

    print("[INFO]: loaded IMU data")

    # load cam data
    camera_sensors = []
    print("[INFO]: loading camera data...")
    for i in range(len(args.cam)):
        cam = camera_sensor(f"cam-{i}", args.cam[i])
        cam.set_period(T_cam)
        estimator.add_camera(cam.id, cam.get_properties())
        camera_sensors.append(cam)
    print("[INFO]: loaded camera data")

    # load GPS data
    gps_sensors = []
    print("[INFO]: loading GPS data...")
    for i in range(len(args.gps)):
        gps = GPS_sensor(f"gps-{i}", args.gps[i])
        gps.set_period(T_gps)
        estimator.add_GPS(gps.id, gps.get_properties())
        gps_sensors.append(gps)
    print("[INFO]: loaded GPS data")

    # allocate data buffer
    pose_err = np.zeros(6)
    pose_err_all = np.zeros([N, 6])

    covariance = np.eye(9) * 1e-6
    covariance_all  = np.zeros([9, 9, N])
    covariance_all[:, :, 0] = np.copy(covariance)

    pos = r_vi_i_all[0, :]
    pos_est_all = np.zeros_like(r_vi_i_all)
    pos_est_all[0, :]  = np.copy(pos)

    #################
    # run algorithm #
    #################

    # run simulation
    time_prev = 0
    for k in range(N):
        for imu in imu_sensors:
            measurement = imu.get_measurement(k)
            if measurement == None:
                continue
            time_now = k * dt
            time_diff = time_now - time_prev
            estimator.handle_IMU_measurement(imu.id, time_diff, measurement[0], measurement[1])
            time_prev = time_now

        for cam in camera_sensors:
            measurement = cam.get_measurement(k)
            if measurement == None:
                continue
            estimator.handle_camera_measurement(cam.id, measurement[0], measurement[1])

        for gps in gps_sensors:
            measurement = gps.get_measurement(k)
            if measurement == None:
                continue
            estimator.handle_GPS_measurement(gps.id, measurement)

        # get current state estimate and uncertainty
        C_iv, vel, r_vi_i = estimator.get_state_estimate()
        covariance = estimator.get_state_covariance()

        # compute error
        C_err    = np.linalg.inv(C_iv) @ C_iv_all[:, :, k]
        pose_err[0: 3] = Rotation.from_matrix(C_err).as_rotvec()
        pose_err[3: 6] = r_vi_i - r_vi_i_all[k, :]
        pose_err_all[k, :] = np.copy(pose_err)
        covariance_all[:, :, k] = np.copy(covariance[0: 9, 0: 9])
        pos_est_all[k, :] = np.copy(r_vi_i)

    ################
    # plot results #
    ################
    # plot trajectory
    plot_trajectory(r_vi_i_all[0: N, :], pos_est_all[0: N, :])

    # plot state error
    plot_state_error_and_uncertainty(pose_err_all[0: N, 0: 3], covariance_all[0: 3, 0: 3, 0: N], titles=["e_roll", "e_pitch", "e_yaw"])
    plot_state_error_and_uncertainty(pose_err_all[0: N, 3: 6], covariance_all[6: 9, 6: 9, 0: N], titles=["e_pos_x", "e_pos_y", "e_pos_z"])

    print("[INFO]: done")


if __name__ == "__main__":
    main()