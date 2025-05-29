clear all
close all
clc

load hs_torque_data.mat

angle_flat = angle_mat(torque_partial ~= 0);
elev_flat = elev_mat(torque_partial ~= 0);
hs_torque_partial_flat = torque_partial(torque_partial ~= 0);

