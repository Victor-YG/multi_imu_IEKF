clear all
clc

% IMU calibration info
C_sv   = eye(3);
r_sv_v = zeros(1, 3);
r_sv_v = [0, 0, 0];

% create imu
imu = imuSensor;
fn = fullfile('mpu6050.json');
loadparams(imu, fn, 'mpu6050_6axis_calibration');

imu.Gyroscope
imu.Accelerometer

w_omega = imu.Gyroscope.RandomWalk;
w_accel = imu.Accelerometer.RandomWalk;
n_omega = imu.Gyroscope.NoiseDensity;
n_accel = imu.Accelerometer.NoiseDensity;

% load trajectory data
load("data_trajectory.mat")

accel_s_all   = zeros(N, 3);
y_accel_all   = zeros(N, 3);
y_omega_all   = zeros(N, 3);
y_accel_i_all = zeros(N, 3);

for k = 1 : N
    C_iv    = C_iv_all(:, :, k);
    alpha_v = alpha_v_all(k, :);
    omega_v = omega_v_all(k, :);
    omega_i = omega_i_all(k, :);
    accel_i = accel_i_all(k, :);

    C_vi = inv(C_iv);
    C_si = C_sv * C_vi;
    
    % compute acceleration and omega at IMU location (inertial frame)
    accel_t = cross(alpha_v, r_sv_v);
    accel_r = cross(omega_v, cross(omega_v, r_sv_v));
    accel_i = accel_i + (C_iv * (accel_t + accel_r)')';

    % compute acceleration in sensor frame (for reference)
    accel_s = (C_si * accel_i')';
    accel_s_all(k, :) = accel_s;

    % take measurement
    % negative sign is needed as the IMU model output negative accel measurement
    [y_accel, y_omega] = imu(-accel_i, omega_i, C_si);

    y_accel_all(k, :) = y_accel;
    y_omega_all(k, :) = y_omega;

    % store transformed measurement in inertial frame with gravity removed
    y_accel_i = C_iv * y_accel' - [0; 0; 9.81];
    y_accel_i_all(k, :) = y_accel_i';
end

save("data_imu.mat", "C_sv", "r_sv_v", "w_omega", "w_accel", "n_omega", "n_accel", ...
                     "accel_s_all", "y_omega_all", "y_accel_all", "y_accel_i_all");

disp("done generating imu measurement");