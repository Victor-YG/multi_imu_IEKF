'''This script evaluates different multi-IMU algorithm using project aria everyday activities dataset'''

import os
import argparse

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from projectaria_tools.core.sophus import SO3, SE3, interpolate, iterativeMean

from core.metrics import *
from core.VIMU_estimator import *

sample_folder = "/media/victor/T7/datasets/aria/loc1_script1_seq1_rec1"

n_omega_l = 8.73e-5 # (rad/s) / sqrt(Hz)
n_accel_l = 8.83e-4 # (m/s^2) / sqrt(Hz)
# w_omega_l = 0.001 # dps
# w_accel_l = 0.000028 # g

n_omega_r = 1.75e-4 # (rad/s) / sqrt(Hz)
n_accel_r = 7.85e-4 # (m/s^2) / sqrt(Hz)
# w_omega_r = 0.0013 # dps
# w_accel_r = 0.000035 # g


def load_and_config_data_provider(folder):
    # create data provider
    vrs_file = f"{folder}/recording.vrs"
    provider = data_provider.create_vrs_data_provider(vrs_file)

    # get stream id
    SID_IMU_L = provider.get_stream_id_from_label("imu-left")  # 1202-2
    SID_IMU_R = provider.get_stream_id_from_label("imu-right") # 1202-1
    SID_CAM_SLAM_L = provider.get_stream_id_from_label("camera-slam-left")  # 1201-1
    SID_CAM_SLAM_R = provider.get_stream_id_from_label("camera-slam-right") # 1201-2

    # starts by default options and only activate imu and slam camera
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(SID_IMU_L)
    deliver_option.activate_stream(SID_IMU_R)
    deliver_option.activate_stream(SID_CAM_SLAM_L)
    deliver_option.activate_stream(SID_CAM_SLAM_R)

    return provider, deliver_option


def load_closed_loop_trajectory(folder, k0=0):
    closed_loop_path = f"{folder}/mps/slam/closed_loop_trajectory.csv"
    trajectory = mps.read_closed_loop_trajectory(closed_loop_path)
    t0 = trajectory[k0].tracking_timestamp.seconds * 1e9 + trajectory[k0].tracking_timestamp.microseconds * 1e3

    # get all trajectory waypoints
    N = len(trajectory)
    ref_C_ib_all = np.zeros([3, 3, N])
    ref_r_bi_i_all = np.zeros([N, 3])
    for i in range(N):
        T_ib = trajectory[i].transform_world_device.to_matrix()
        C_ib = np.copy(T_ib[0: 3, 0: 3])
        r_bi_i = np.copy(T_ib[0: 3, 3])
        ref_C_ib_all[:, :, i] = C_ib
        ref_r_bi_i_all[i, :]  = r_bi_i

    v0 = np.array(trajectory[k0].device_linear_velocity_device)
    omega_0 = np.array(trajectory[k0].angular_velocity_device)

    # # plot closed-loop trajectory
    # plot_trajectory_and_initial_velocity(ref_r_bi_i_all, v0)

    return ref_C_ib_all[:, :, k0:], ref_r_bi_i_all[k0: ], v0, omega_0, t0, N


def load_points_filtered_by_confidence(folder, inverse_distance_std_threshold=0.001, distance_std_threshold=0.0001):
    global_points_path = f"{folder}/mps/slam/semidense_points.csv.gz"
    points = mps.read_global_point_cloud(global_points_path)
    filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)

    # # for point in filtered_points:
    #     print(point.position_world)
    # exit()

    # # plot filtered points
    # M = len(filtered_points)
    # pt = np.zeros([M, 3])

    # for i in range(M):
    #     pt[i, 0] = filtered_points[i].position_world[0]
    #     pt[i, 1] = filtered_points[i].position_world[1]
    #     pt[i, 2] = filtered_points[i].position_world[2]
    # fig = plt.scatter(pt[:, 0], pt[:, 1], c='b')
    # plt.show()

    # fig = plt.scatter(ref_r_bi_i_all[:, 0], ref_r_bi_i_all[:, 1], c='g')
    # plt.show()

    print(f"[INFO]: loaded {len(filtered_points)} points.")
    return filtered_points


def load_imu(provider, label):
    imu = provider.get_device_calibration().get_imu_calib(label)
    T_bs = np.copy(imu.get_transform_device_imu().to_matrix())
    # bias_omega = imu.raw_to_rectified_gyro(np.zeros(3))
    # bias_accel = imu.raw_to_rectified_accel(np.zeros(3))
    return imu, T_bs


def load_camera(provider, label):
    cam_calib = provider.get_device_calibration().get_camera_calib(label)
    T_bc = np.copy(cam_calib.get_transform_device_camera().to_matrix())
    return cam_calib, T_bc


def get_imu_data(imu, data):
    imu_data = data.imu_data()
    omega = np.array(imu_data.gyro_radsec)
    accel = np.array(imu_data.accel_msec2)
    omega_rect = imu.raw_to_rectified_gyro(omega)
    accel_rect = imu.raw_to_rectified_accel(accel)
    omega_bias = omega - omega_rect
    accel_bias = accel - accel_rect
    return omega_rect, accel_rect
    # return omega, accel


def gen_cam_data(camera, T_ci, points):
    pt_3d = []
    kp_2d = []

    for point in points:
        pt_i = np.ones(4)
        pt_i[0] =point.position_world[0]
        pt_i[1] =point.position_world[1]
        pt_i[2] =point.position_world[2]
        pt_c = T_ci @ pt_i

        kp = camera.project(pt_c[0: 3])
        if kp is None:
            continue

        if camera.is_visible(kp):
            pt_3d.append(pt_i[0: 3])
            kp_2d.append(kp)

    if len(pt_3d) == 0:
        return None, None

    return np.stack(pt_3d, axis=0), np.stack(kp_2d, axis=0)


def sim_cam_data(camera, T_ci, M=10):
    pt_3d = []
    kp_2d = []

    w, h   = camera.get_image_size()
    fx, fy = camera.get_focal_lengths()
    cx, cy = camera.get_principal_point()

    for m in range(M):
        u = np.random.rand() * w
        v = np.random.rand() * h
        d = (np.random.rand() + 0.2) * 10.0

        # reproject to 3D
        r_pc_c = np.zeros(4)
        r_pc_c[0] = d * (u - cx) / fx
        r_pc_c[1] = d * (v - cy) / fy
        r_pc_c[2] = d
        r_pc_c[3] = 1.0

        # transform to inertial frame
        r_pi_i = np.linalg.inv(T_ci) @ r_pc_c

        pt_3d.append(r_pi_i[0: 3])
        kp_2d.append(np.array([u, v]))

    return np.stack(pt_3d, axis=0), np.stack(kp_2d, axis=0)


def sim_point_data(T_iv, M=10):
    pt_i = []
    pt_v = []

    for m in range(M):
        p_v = np.ones(4)
        p_v[0] = (np.random.rand() - 0.5) * 10.0
        p_v[1] = (np.random.rand() - 0.5) * 10.0
        p_v[2] = (np.random.rand() - 0.5) * 10.0
        p_i = T_iv @ p_v
        pt_v.append(p_v)
        pt_i.append(p_i)

    return np.stack(pt_i, axis=0), np.stack(pt_v, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aea",  help="Path to Aria Everyday Activities data folder", default=sample_folder, required=False)
    parser.add_argument("--algo", help="Algorithm for state estimation (SIMU, VIMU)", default="SIMU", required=False)
    # parser.add_argument("--out",  help="Output folder to dump evaluation result", default=None, required=True)
    args = parser.parse_args()

    estimator = None
    imu_l = None
    imu_r = None
    camera = None

    #############
    # load data #
    #############

    provider, deliver_option = load_and_config_data_provider(args.aea)

    # load closed-loop trajectory
    ref_C_ib_all, ref_r_bi_i_all, v0, omega_0, t0, N = load_closed_loop_trajectory(args.aea, 2000)
    T_ib_0 = rotation_and_translation_to_pose(ref_C_ib_all[:, :, 0], ref_r_bi_i_all[0, :])

    # # read points and observation from mps
    # points = load_points_filtered_by_confidence(args.aea)

    # get camera extrinsic calibration
    cam_calib_l, T_bc_l = load_camera(provider, "camera-slam-left")
    cam_calib_r, T_bc_r = load_camera(provider, "camera-slam-right")

    # setup camera simulator and get intrinsics
    fx = 150
    img_w = 512
    img_h = 512
    camera = calibration.get_linear_camera_calibration(img_w, img_h, fx)
    cx, cy = camera.get_principal_point()
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])

    # TODO::single IMU for now; update implementation for VIMU
    if args.algo == "SIMU":
        # get imu extrinsic calibration
        imu, T_bs = load_imu(provider, "imu-left")
        # imu, T_bs = load_imu(provider, "imu-right")

        # single IMU vehicle frame is at IMU sensor frame
        T_bv = T_bs

        # compute initial velocity (inertial frame)
        r_vb_b = T_bv[0: 3, 3]
        v0_v_i = ref_C_ib_all[:, :, 0] @ (v0 + np.cross(omega_0, r_vb_b))

        T_vb = np.linalg.inv(T_bv)
        T_cv_l = np.linalg.inv(T_bc_l) @ T_bv
        T_cv_r = np.linalg.inv(T_bc_r) @ T_bv
        T_iv_0 = T_ib_0 @ T_bv

        estimator = VIMU_estimator(T_iv_0, v0_v_i, pre_proc="None")
        estimator.add_IMU("imu-left", IMU_sensor_prop(np.eye(4), n_omega_l * np.ones(3), n_accel_l * np.ones(3), 0.1 * np.ones(3), 0.1 * np.ones(3)))
        # estimator.add_IMU("imu-right", IMU_sensor_prop(np.eye(4), n_omega_r * np.ones(3), n_accel_r * np.ones(3), 0.1 * np.ones(3), 0.1 * np.ones(3)))
    elif args.algo == "VIMU":
        pass

    estimator.add_camera("camera-slam-left",  camera_sensor_prop(T_cv_l, K, 0.1, 0.1))
    estimator.add_camera("camera-slam-right", camera_sensor_prop(T_cv_r, K, 0.1, 0.1))

    #################
    # run algorithm #
    #################

    est_C_ib_all   = np.zeros_like(ref_C_ib_all)
    est_r_bi_i_all = np.zeros_like(ref_r_bi_i_all)
    est_C_iv, est_r_vi_i = estimator.get_state_estimate()
    est_T_iv = rotation_and_translation_to_pose(est_C_iv, est_r_vi_i)
    est_C_ib_all[:, :, 0] = est_T_iv[0: 3, 0: 3]
    est_r_bi_i_all[0, :]  = est_T_iv[0: 3, 3]
    pose_err = np.zeros(6)
    pose_err_all = np.zeros([N, 6])
    covariance_all = np.zeros([9, 9, N])
    est_T_ib = est_T_iv @ T_vb

    # Async iterator to deliver sensor data for all streams in device time order
    for data in provider.deliver_queued_sensor_data(deliver_option):
        # get device time
        t_curr = data.get_time_ns(TimeDomain.DEVICE_TIME)
        k = int(np.floor((t_curr - t0) / 1e6)) # timestep in ms

        if t_curr - t0 < 0:
            continue

        # get sensor label
        sid = data.stream_id()
        label = provider.get_label_from_stream_id(sid)

        if label == "imu-left" or label == "imu-right":
            time = (t_curr - t0) / 1e9 # ns to sec
            omega, accel = get_imu_data(imu, data)
            estimator.handle_IMU_measurement(label, time, omega, accel)

        elif label == "camera-slam-left" or label == "camera-slam-right":
            C_ib = ref_C_ib_all[:, :, k]
            r_bi_i = ref_r_bi_i_all[k, :]
            T_ib = rotation_and_translation_to_pose(C_ib, r_bi_i)

            # T_cv = estimator.camera[label].T_cv
            # T_ci = T_cv @ T_vb @ np.linalg.inv(T_ib)
            # # print(T_ci)
            # # pt_3d, kp_2d = gen_cam_data(camera, T_ci, points)
            # pt_3d, kp_2d = sim_cam_data(camera, T_ci, 400)
            # estimator.handle_camera_measurement(label, pt_3d, kp_2d)

            T_iv = T_ib @ T_bv
            pt_i, pt_v = sim_point_data(T_iv, 10)
            estimator.handle_point_measurement(pt_i, pt_v)
        else:
            continue
            print("[FAIL]: unexpected sensor data received.")

        C_iv, r_vi_i = estimator.get_state_estimate()
        T_iv = rotation_and_translation_to_pose(C_iv, r_vi_i)
        covariance_all[:, :, k] = estimator.get_state_covariance()[0: 9, 0: 9]
        est_T_ib = T_iv @ T_vb
        est_C_ib = est_T_ib[0: 3, 0: 3]
        est_r_bi_i = est_T_ib[0: 3, 3]
        est_C_ib_all[:, :, k] = np.copy(est_C_ib)
        est_r_bi_i_all[k, :] = np.copy(est_r_bi_i)
        C_err = np.linalg.inv(est_C_ib) @ ref_C_ib_all[:, :, k]
        pose_err[0: 3] = Rotation.from_matrix(C_err).as_rotvec()
        pose_err[3: 6] = est_r_bi_i - ref_r_bi_i_all[k, :]
        pose_err_all[k, :] = np.copy(pose_err)

        # print(f"[INFO]: t = {t_curr}, k = {k}, dt = {dt}; received data from {label}, r_vi_i = {est_r_vi_i}")

        if np.linalg.norm(pose_err[3: 6]) > 1:
            print(f"[WARN]: trajectory diverged from ground truth at k = {k}")
            break

        # if k > 3000:
        #     break

    ################
    # plot results #
    ################

    for i in range(k - 1):
        if np.linalg.norm(est_r_bi_i_all[i, :]) == 0.0:
            est_r_bi_i_all[i, :] = 0.5 * (est_r_bi_i_all[i - 1, :] + est_r_bi_i_all[i + 1, :])
            pose_err_all[i, :] = 0.5 * (pose_err_all[i - 1, :] + pose_err_all[i + 1, :])

    mae, mse, rmse = compute_metrics(pose_err_all[0: k - 1, :])
    np.set_printoptions(precision=8)
    print(f"[INFO]:  mae = { mae}")
    print(f"[INFO]: rmse = {rmse}")
    print(f"[INFO]:  mse = { mse}")

    plot_trajectory(ref_r_bi_i_all[0: k - 1, :], est_r_bi_i_all[0: k - 1, :])

    # plot state error
    plot_state_error_and_uncertainty(pose_err_all[0: k - 1, 0: 3], covariance_all[0: 3, 0: 3, 0: k - 1], titles=["e_roll", "e_pitch", "e_yaw"])
    plot_state_error_and_uncertainty(pose_err_all[0: k - 1, 3: 6], covariance_all[6: 9, 6: 9, 0: k - 1], titles=["e_pos_x", "e_pos_y", "e_pos_z"])

    print("[INFO]: Done")


if __name__ == "__main__":
    main()