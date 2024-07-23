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

from core.VIMU_estimator import *

sample_folder = "../../datasets/aria/loc1_script1_seq1_rec1"

# n_omega_l = 8.73e-5 # (rad/s) / sqrt(Hz)
# n_accel_l = 8.83e-4 # (m/s^2) / sqrt(Hz)
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

    N = len(trajectory) - k0
    ref_C_iv_all = np.zeros([3, 3, N])
    ref_r_vi_i_all = np.zeros([N, 3])
    for i in range(N):
        T_iv = trajectory[i].transform_world_device.to_matrix()
        C_iv = np.copy(T_iv[0: 3, 0: 3])
        r_vi_i = np.copy(T_iv[0: 3, 3])
        ref_C_iv_all[:, :, i] = C_iv
        ref_r_vi_i_all[i, :]  = r_vi_i

    v0 = np.array(trajectory[k0].device_linear_velocity_device)
    omega_0 = np.array(trajectory[k0].angular_velocity_device)
    return ref_C_iv_all[:, :, k0:], ref_r_vi_i_all[k0: ], v0, omega_0, t0, N


def load_points_filtered_by_confidence(folder, inverse_distance_std_threshold=0.001, distance_std_threshold=0.0001):
    global_points_path = f"{folder}/mps/slam/semidense_points.csv.gz"
    points = mps.read_global_point_cloud(global_points_path)
    filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)

    # for point in filtered_points:
    #     print(point.position_world)
    # exit()

    print(f"[INFO]: loaded {len(filtered_points)} points.")
    return filtered_points


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aea",  help="Path to Aria Everyday Activities data folder", default=sample_folder, required=False)
    parser.add_argument("--algo", help="State estimation algorithm to evaluate (VIMU or CM-IMU)", default="VIMU")
    # parser.add_argument("--out",  help="Output folder to dump evaluation result", default=None, required=True)
    args = parser.parse_args()

    estimator = None
    imu_l = None
    imu_r = None
    camera = None

    #############
    # load data #
    #############

    args.aea = sample_folder
    provider, deliver_option = load_and_config_data_provider(args.aea)

    # load closed-loop trajectory
    ref_C_iv_all, ref_r_vi_i_all, v0, omega_0, t0, N = load_closed_loop_trajectory(args.aea, 1000)

    # read points and observation from mps
    points = load_points_filtered_by_confidence(args.aea)

    # get imu extrinsic calibration
    imu_l = provider.get_device_calibration().get_imu_calib("imu-left")
    T_vs_l = np.copy(imu_l.get_transform_device_imu().to_matrix())
    r_sv_v_l = T_vs_l[0: 3, 3]
    T_sv_l = np.linalg.inv(T_vs_l)
    bias_omega_l = imu_l.raw_to_rectified_gyro(np.zeros(3))
    bias_accel_l = imu_l.raw_to_rectified_accel(np.zeros(3))
    imu_r = provider.get_device_calibration().get_imu_calib("imu-right")
    T_vs_r = np.copy(imu_r.get_transform_device_imu().to_matrix())
    T_sv_r = np.linalg.inv(T_vs_r)
    r_sv_v_r = T_vs_r[0: 3, 3]

    # choose left vs right IMU
    imu = imu_r
    T_sv = T_sv_r
    r_sv_v = r_sv_v_r

    # compute initial velocity (inertial frame)
    v0_s_i = ref_C_iv_all[:, :, 0] @ (v0 + np.cross(omega_0, r_sv_v))

    # get camera extrinsic calibration
    cam_calib = provider.get_device_calibration().get_camera_calib("camera-slam-left")
    T_vc_l = np.copy(cam_calib.get_transform_device_camera().to_matrix())
    T_cv_l = np.linalg.inv(T_vc_l)

    cam_calib = provider.get_device_calibration().get_camera_calib("camera-slam-right")
    T_vc_r = np.copy(cam_calib.get_transform_device_camera().to_matrix())
    T_cv_r = np.linalg.inv(T_vc_r)

    fx = 150
    img_w = 512
    img_h = 512
    camera = calibration.get_linear_camera_calibration(img_w, img_h, fx)
    cx, cy = camera.get_principal_point()

    # TODO::single IMU for now; update implementation for VIMU
    T_iv_0 = rotation_and_translation_to_pose(ref_C_iv_all[:, :, 0], ref_r_vi_i_all[0, :])
    T_is_0 = T_iv_0 @ T_vs_r
    estimator = VIMU_estimator(T_is_0, v0_s_i)
    # estimator.add_IMU("imu-left", IMU_sensor_prop(np.eye(4), n_omega_l * np.ones(3), n_accel_l * np.ones(3), 0.1 * np.ones(3), 0.1 * np.ones(3)))
    # estimator.add_IMU("imu-right", IMU_sensor_prop(np.eye(4), n_omega_r * np.ones(3), n_accel_r * np.ones(3), 0.1 * np.ones(3), 0.1 * np.ones(3)))
    estimator.add_IMU("imu-right", IMU_sensor_prop(np.eye(4), 0.1 * np.ones(3), 0.2 * np.ones(3), 0.1 * np.ones(3), 0.1 * np.ones(3)))
    T_cs_l = T_cv_l @ T_vs_r
    T_cs_r = T_cv_r @ T_vs_r
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    # estimator.add_camera("camera-slam-left",  camera_sensor_prop(T_cs_l, K, 0.000001, 0.000001))
    # estimator.add_camera("camera-slam-right", camera_sensor_prop(T_cs_r, K, 0.000001, 0.000001))
    estimator.add_camera("camera-slam-left",  camera_sensor_prop(T_cs_l, K, 0.1, 0.1))
    estimator.add_camera("camera-slam-right", camera_sensor_prop(T_cs_r, K, 0.1, 0.1))
    # estimator.add_camera("camera-slam-left",  camera_sensor_prop(np.eye(4), K, 0.1, 0.1))
    # estimator.add_camera("camera-slam-right", camera_sensor_prop(np.eye(4), K, 0.1, 0.1))

    # if args.algo == "VIMU":
    #     # create virtual IMU (reference frame denoted as z)
    #     r_sv_v_l = T_vs_l[0: 3, 3]
    #     r_sv_v_r = T_vs_r[0: 3, 3]
    #     r_zv_v = 0.5 * (r_sv_v_l + r_sv_v_r)
    #     T_vz = np.eye(4)
    #     T_vz[0: 3, 3] = r_zv_v
    #
    #     T_iz_0 = T_iv_0 @ T_vz
    #     estimator = VIMU_estimator(T_iz_0)
    #
    #     # add IMUs
    #     T_sz_l = T_sv_l @ T_vz
    #     T_sz_r = T_sv_r @ T_vz
    #     estimator.add_IMU("imu-left",  IMU_sensor_prop(T_sz_l, 0.1 * np.ones(3), 0.5 * np.ones(3), 0.01 * np.ones(3), 0.01 * np.ones(3)))
    #     estimator.add_IMU("imu-right", IMU_sensor_prop(T_sz_r, 0.1 * np.ones(3), 0.5 * np.ones(3), 0.01 * np.ones(3), 0.01 * np.ones(3)))
    #
    #     # add camera
    #     T_cz_l = T_cv_l @ T_vz
    #     T_cz_r = T_cv_r @ T_vz
    #     K = np.array([[fx, 0.0, 0.0], [0.0, fx, 0.0], [0.0, 0.0, 1.0]])
    #     estimator.add_camera("camera-slam-left",  camera_sensor_prop(T_cz_l, K, 0.01, 0.01))
    #     estimator.add_camera("camera-slam-right", camera_sensor_prop(T_cz_r, K, 0.01, 0.01))

    #################
    # run algorithm #
    #################

    est_C_iv_all = np.zeros_like(ref_C_iv_all)
    est_r_vi_i_all = np.zeros_like(ref_r_vi_i_all)
    C_is_est, _, r_si_i_est = estimator.get_state_estimate()
    T_is_est = rotation_and_translation_to_pose(C_is_est, r_si_i_est)
    T_iv_est = T_is_est @ T_sv
    est_C_iv_all[:, :, 0] = T_iv_est[0: 3, 0: 3]
    est_r_vi_i_all[0, :] = T_iv_est[0: 3, 3]
    # est_C_iv_all[:, :, 0] = np.copy(ref_C_iv_all[:, :, 0])
    # est_r_vi_i_all[0, :] = ref_r_vi_i_all[0, :]

    # Async iterator to deliver sensor data for all streams in device time order
    t_prev = None
    dt = 0.0
    done = False
    for data in provider.deliver_queued_sensor_data(deliver_option):
        # get device time
        t_curr = data.get_time_ns(TimeDomain.DEVICE_TIME)
        k = int(np.floor((t_curr - t0) / 1e6)) # timestep in ms

        if t_curr - t0 < 0:
            continue

        # get sensor label
        sid = data.stream_id()
        label = provider.get_label_from_stream_id(sid)

        # if label == "imu-left":
        if label == "imu-right":
            if t_prev == None:
                t_prev = t_curr
                continue

            dt = (t_curr - t_prev) / 1e9 # ns to sec
            t_prev = t_curr

            omega, accel = get_imu_data(imu, data)
            estimator.handle_IMU_measurement(label, dt, omega, accel)

        elif label == "camera-slam-left" or label == "camera-slam-right":
            C_iv = ref_C_iv_all[:, :, k]
            r_vi_i = ref_r_vi_i_all[k, :]
            T_iv = rotation_and_translation_to_pose(C_iv, r_vi_i)
            T_cs = estimator.camera[label].T_cv
            T_ci = T_cs @ T_sv @ np.linalg.inv(T_iv)
            # print(T_ci)

            # pt_3d, kp_2d = gen_cam_data(camera, T_ci, points)
            pt_3d, kp_2d = sim_cam_data(camera, T_ci, 400)
            if pt_3d is None:
                print("[WARN]: no visible landmark detected")
                continue
            # else:
                # print(f"[INFO]: detected {pt_3d.shape[0]} landmarks")

            estimator.handle_camera_measurement(label, pt_3d, kp_2d)
            done = True

        else:
            continue
            print("[FAIL]: unexpected sensor data received.")

        C_is, __, r_si_i = estimator.get_state_estimate()
        T_is = rotation_and_translation_to_pose(C_is, r_si_i)
        T_iv_est = T_is @ T_sv
        est_C_iv_all[:, :, k] = np.copy(T_iv_est[0: 3, 0: 3])
        r_vi_i_est = np.copy(T_iv_est[0: 3, 3])
        est_r_vi_i_all[k, :] = r_vi_i_est
        # print(f"[INFO]: t = {t_curr}, k = {k}, dt = {dt}; received data from {label}, r_vi_i = {r_vi_i_est}")
        r_vi_i_err = r_vi_i_est - ref_r_vi_i_all[k, :]
        if np.linalg.norm(r_vi_i_err) > 1:
            print(f"[WARN]: trajectory diverged from ground truth at k = {k}")
            break

        # if k < 30000:
        #     done = False
        # else:
        #     done = True

        # if done:
        #     break

    ################
    # plot results #
    ################
    #TODO::plot closed-loop trajectory
    # # ref_r_vi_i_all = ref_r_vi_i_all[10000: 20000, :]
    # plot_trajectory(ref_r_vi_i_all, np.zeros_like(ref_r_vi_i_all))

    for i in range(k - 1):
        if np.linalg.norm(est_r_vi_i_all[i, :]) == 0.0:
            est_r_vi_i_all[i, :] = 0.5 * (est_r_vi_i_all[i - 1, :] + est_r_vi_i_all[i + 1, :])

    # plot_trajectory(ref_r_vi_i_all, est_r_vi_i_all)
    plot_trajectory(ref_r_vi_i_all[0: k - 1, :], est_r_vi_i_all[0: k - 1, :])

    # # plot filtered points
    # M = len(points)
    # pt = np.zeros([M, 3])

    # for i in range(M):
    #     pt[i, 0] = points[i].position_world[0]
    #     pt[i, 1] = points[i].position_world[1]
    #     pt[i, 2] = points[i].position_world[2]
    # fig = plt.scatter(pt[:, 0], pt[:, 1], c='b')
    # plt.show()

    # fig = plt.scatter(ref_r_vi_i_all[:, 0], ref_r_vi_i_all[:, 1], c='g')
    # plt.show()


    print("[INFO]: Done")


if __name__ == "__main__":
    main()