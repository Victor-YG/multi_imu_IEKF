import scipy
import numpy as np

from core.utils import *


#####################
# sensor properties #
#####################

class IMU_sensor_prop(object):
    def __init__(self, T_sv, n_omega, n_accel, w_omega, w_accel):
        self.T_sv = T_sv
        self.n_omega = n_omega
        self.n_accel = n_accel
        self.w_omega = w_omega
        self.w_accel = w_accel


class camera_sensor_prop(object):
    def __init__(self, T_cv, K, var_n_u, var_n_v):
        self.T_cv = T_cv
        self.K = K
        self.var_n_u = var_n_u
        self.var_n_v = var_n_v


class GPS_sensor_prop(object):
    def __init__(self, T_sv, var_n_x, var_n_y, var_n_z):
        self.T_sv = T_sv
        self.var_n_x = var_n_x
        self.var_n_y = var_n_y
        self.var_n_z = var_n_z


####################
# simulated sensor #
####################

class IMU_sensor(object):
    def __init__(self, id, filename):
        '''construct an IMU object with the input .dat file'''

        self.id = id
        dataset = scipy.io.loadmat(filename)
        self.period = 1

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
        self.T_sv        = self.get_transformation()

        print(f"[INFO]: loaded  IMU data from '{filename}'")


    def set_period(self, period):
        '''sample period in ms'''
        self.period = period


    def get_id(self):
        return self.id


    def get_transformation(self):
        '''get the transformation from vehicle frame to sensor frame (T_sv)'''

        return Cr2T(self.C_sv, self.r_sv_v)


    def get_noise(self):
        '''
        get the noise magnitude of the IMU
            - n_omega is in unit of (rad/s) / sqrt(Hz)
            - n_accel is in unit of (m/s^2) / sqrt(Hz)
        '''
        return self.n_omega, self.n_accel


    def get_bias_drift(self):
        '''
        get the noise magnitude of the IMU
            - w_omega is in unit of (rad/s) * sqrt(Hz)
            - w_accel is in unit of (m/s^2) * sqrt(Hz)
        '''
        return self.w_omega, self.w_accel


    def get_properties(self):
        '''get the property for initializing estimator'''

        return IMU_sensor_prop(self.get_transformation(), self.n_omega, self.n_accel, self.w_omega, self.w_accel)


    def get_measurement(self, k):
        '''
        get the measurement of the IMU at the given timestamp k if available
            - y_omega is in unit of (rad/s)
            - y_accel is in unit of (m/s^2)
        '''

        if k % self.period != 0:
            return None

        return self.y_omega_all[k, :], self.y_accel_all[k, :]


class VIMU_sensor(IMU_sensor):
    def __init__(self, id, filenames):
        '''virtual IMU that contains multiple IMUs'''

        self.id = id
        self.imu_all = []

        for i in range(len(filenames)):
            imu = IMU_sensor(f"imu-{i}", filenames[i])
            self.imu_all.append(imu)

        self.num_imu = len(self.imu_all)

        # get the centroid of IMUs
        self.avg_r_sv_v = 0.0
        for imu in self.imu_all:
            self.avg_r_sv_v += imu.r_sv_v

        self.avg_r_sv_v = self.avg_r_sv_v / self.num_imu

        # get frame of reference of VIMU
        self.T_sv = self.get_transformation()

        # get noise and bias drift
        self.n_omega, self.n_accel = self.get_noise()
        self.w_omega, self.w_accel = self.get_bias_drift()


    def get_transformation(self):
        '''get the transformation from vehicle frame to sensor frame (T_sv)'''

        return Cr2T(np.eye(3), self.avg_r_sv_v)


    def get_noise(self):
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


    def get_bias_drift(self):
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


    def get_measurement(self, k):
        '''get average measurement (omega, accel)'''

        if k % self.period != 0:
            return None

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


class camera_sensor(object):
    def __init__(self, id, filename):
        '''construct an IMU object with the input .dat file'''

        self.id = id
        dataset = scipy.io.loadmat(filename)

        self.C_vc    = dataset["C_vc"]
        self.C_cv    = np.linalg.inv(self.C_vc)
        self.r_cv_v  = dataset["r_cv_v"]
        self.K       = np.eye(3)
        self.K[0, 0] = float(dataset["fx"][0][0])
        self.K[1, 1] = float(dataset["fy"][0][0])
        self.K[0, 2] = dataset["cx"][0][0]
        self.K[1, 2] = dataset["cy"][0][0]
        self.var_n_u = dataset["var_n_u"]
        self.var_n_v = dataset["var_n_v"]
        self.obs_all = dataset["obs_all"]
        self.M       = int(dataset["M"][0][0])
        self.T_cv    = self.get_transformation()

        print(f"[INFO]: loaded camera data from '{filename}'")


    def set_period(self, period):
        '''sample period in ms'''
        self.period = period


    def get_transformation(self):
        return Cr2T(self.C_cv, self.r_cv_v)


    def get_noise(self):
        return self.var_n_u, self.var_n_v


    def get_properties(self):
        '''get the property for initializing estimator'''

        return camera_sensor_prop(self.T_cv, self.K, self.var_n_u, self.var_n_v)


    def get_measurement(self, k):
        '''get all landmark (pt_3d) and image feature (kp_2d) at timestep k'''

        if k % self.period != 0:
            return None

        pt_3d = np.zeros([self.M, 3])
        kp_2d = np.zeros([self.M, 2])

        for m in range(self.M):
            [x, y, z, u, v, n_u, n_v] = self.obs_all[k, m, :].flatten().tolist()
            u += n_u
            v += n_v
            pt_3d[m, 0] = x
            pt_3d[m, 1] = y
            pt_3d[m, 2] = z
            kp_2d[m, 0] = u
            kp_2d[m, 1] = v

        return pt_3d, kp_2d


class GPS_sensor(object):
    def __init__(self, id, filename):
        '''construct an GPS object with the input .dat file'''

        self.id = id
        dataset = scipy.io.loadmat(filename)

        self.var_n_x     = dataset["var_n_x"][0][0]
        self.var_n_y     = dataset["var_n_y"][0][0]
        self.var_n_z     = dataset["var_n_z"][0][0]
        self.pos_obs_all = dataset["pos_obs_all"]

        print(f"[INFO]: loaded GPS data from '{filename}'")


    def set_period(self, period):
        '''sample period in ms'''
        self.period = period


    def get_transformation(self):
        return np.eye(4)


    def get_noise(self):
        return self.var_n_x, self.var_n_y, self.var_n_z


    def get_properties(self):
        '''get the property for initializing estimator'''

        return GPS_sensor_prop(self.T_sv, self.var_n_x, self.var_n_y, self.var_n_z)


    def get_measurement(self, k):
        '''get GPS measurement at timestep k'''

        if k % self.period != 0:
            return None

        return self.pos_obs_all[k, :]