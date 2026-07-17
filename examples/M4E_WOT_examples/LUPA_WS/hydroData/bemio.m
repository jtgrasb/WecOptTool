clc; clear all; close all;

%% hydro data
hydro = struct();
hydro = readCAPYTAINE(hydro,'lupa_hs_centered.nc'); % hs included in nc file
hydro = radiationIRF(hydro,60,[],[],[],[]);
hydro = radiationIRFSS(hydro,[],[]);
hydro = excitationIRF(hydro,157,[],[],[],[]);

writeBEMIOH5(hydro)

%% Plot hydro data
% plotBEMIO(hydro)
