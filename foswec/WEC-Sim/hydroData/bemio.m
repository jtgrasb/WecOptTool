% clc; clear all; close all;

%% hydro data
hydro = struct();
hydro = readCAPYTAINE(hydro,'foswec.nc','hs');
hydro = radiationIRF(hydro,60,[],[],[],[]);
% hydro = radiationIRFSS(hydro,[],[]);
hydro = excitationIRF(hydro,157,[],[],[],[]);
% writeBEMIOH5(hydro)

% set only the values for the pitch dofs of the flaps
hydro.A(:,:,1) = zeros(size(hydro.A(:,:,1)));
hydro.A(11,11,1) = 0.004;
hydro.A(17,17,1) = 0.004;

%% Plot hydro data
plotBEMIO(hydro)