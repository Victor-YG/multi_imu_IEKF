clear all
clc

load("data_kinematics.mat")

% plot trajectory
figure;
plot3(pos_all(:, 1), pos_all(:, 2), pos_all(:, 3));

% plot velocity
figure;
subplot(3, 1, 1);
plot(vel_all(:, 1));
title('vel_x');

subplot(3, 1, 2);
plot(vel_all(:, 2));
title('vel_y');

subplot(3, 1, 3);
plot(vel_all(:, 3));
title('vel_z');

% plot acceleration
figure;
subplot(3, 1, 1);
plot(accel_all(:, 1));
title('accel_x');

subplot(3, 1, 2);
plot(accel_all(:, 2));
title('accel_y');

subplot(3, 1, 3);
plot(accel_all(:, 3));
title('accel_z');

% plot angular velocity
figure;
subplot(3, 1, 1);
plot(omega_all(:, 1));
title('omega_x');

subplot(3, 1, 2);
plot(omega_all(:, 2));
title('omega_y');

subplot(3, 1, 3);
plot(omega_all(:, 3));
title('omega_z');
