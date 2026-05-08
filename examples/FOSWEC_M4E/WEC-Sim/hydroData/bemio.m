clc; clear all; close all;

%% hydro data
hydro = struct();
hydro = readCAPYTAINE(hydro,'foswec_jeff_hs_submerged.nc'); % hs included in nc file
hydro = radiationIRF(hydro,60,[],[],[],[]);
% hydro = radiationIRFSS(hydro,[],[]);
hydro = excitationIRF(hydro,157,[],[],[],[]);

% add more buoyancy force
% hydro.Vo(2) = 0.04;
% hydro.Vo(3) = 0.04;

writeBEMIOH5(hydro)

%% Plot hydro data
% plotBEMIO(hydro)

%% use foswec 2 stiffness

hydro = struct();
hydro = readCAPYTAINE(hydro,'foswec_jeff_hs.nc'); % hs included in nc file
hydro = radiationIRF(hydro,60,[],[],[],[]);
% hydro = radiationIRFSS(hydro,[],[]);
hydro = excitationIRF(hydro,157,[],[],[],[]);

hydro2 = readH5ToStruct('foswec2.h5');

hydro.Khs(:,:,2) = hydro2.Khs(:,:,1);
hydro.Khs(:,:,3) = hydro2.Khs(:,:,1);
hydro.file = 'foswec_jeff_stiffness2';

writeBEMIOH5(hydro)