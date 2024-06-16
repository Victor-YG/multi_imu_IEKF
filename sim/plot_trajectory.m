clear all
clc

load("data_kinematics.mat")

% plot trajectory
figure;
plot3(r_vi_i_all(:, 1), r_vi_i_all(:, 2), r_vi_i_all(:, 3));

% compute euler angle
euler_all   = zeros(N, 3);
for k = 1 : N
    C_iv = C_iv_all(:, :, k);
    euler = rotm2eul(C_iv);
    euler_all(k, :) = euler;
end

% plot orientation
figure;
subplot(3, 1, 1);
plot(euler_all(:, 1));
title('roll');

subplot(3, 1, 2);
plot(euler_all(:, 2));
title('pitch');

subplot(3, 1, 3);
plot(euler_all(:, 3));
title('yaw');

% plot velocity
figure;
subplot(3, 1, 1);
plot(vel_i_all(:, 1));
title('vel_x');

subplot(3, 1, 2);
plot(vel_i_all(:, 2));
title('vel_y');

subplot(3, 1, 3);
plot(vel_i_all(:, 3));
title('vel_z');

% plot acceleration
figure;
subplot(3, 1, 1);
plot(accel_v_all(:, 1));
title('accel_x');

subplot(3, 1, 2);
plot(accel_v_all(:, 2));
title('accel_y');

subplot(3, 1, 3);
plot(accel_v_all(:, 3));
title('accel_z');

% plot angular velocity
figure;
subplot(3, 1, 1);
plot(omega_v_all(:, 1));
title('omega_x');

subplot(3, 1, 2);
plot(omega_v_all(:, 2));
title('omega_y');

subplot(3, 1, 3);
plot(omega_v_all(:, 3));
title('omega_z');
