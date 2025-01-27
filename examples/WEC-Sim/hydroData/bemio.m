% clc; clear all; close all;

%% hydro data
hydro = struct();
hydro = readCAPYTAINE(hydro,'attenuator2Bod.nc','2_bod');
hydro = radiationIRF(hydro,[],[],[],[],[]);
hydro = radiationIRFSS(hydro,[],[]);
hydro = excitationIRF(hydro,[],[],[],[],[]);
writeBEMIOH5(hydro)

%% Plot hydro data
plotBEMIO(hydro)