% clc; clear all; close all;

%% hydro data
hydro = struct();
hydro = readCAPYTAINE(hydro,'attenuator.nc','1_bod');

% eliminate coupling terms
hydro.A = hydro.A.*eye(6);
hydro.Ainf = hydro.Ainf.*eye(6);
hydro.B = hydro.B.*eye(6);
hydro.Khs = hydro.Khs.*eye(6);

% eliminate non-heave or pitch terms
hydro.A = hydro.A.*[0 0 1 0 1 0];
hydro.Ainf = hydro.Ainf.*[0 0 1 0 1 0];
hydro.B = hydro.B.*[0 0 1 0 1 0];
hydro.Khs = hydro.Khs.*[0 0 1 0 1 0];
hydro.fk_re = hydro.fk_re.*[0 0 1 0 1 0]';
hydro.fk_im = hydro.fk_im.*[0 0 1 0 1 0]';
hydro.fk_ma = hydro.fk_ma.*[0 0 1 0 1 0]';
hydro.fk_ph = hydro.fk_ph.*[0 0 1 0 1 0]';
hydro.sc_re = hydro.sc_re.*[0 0 1 0 1 0]';
hydro.sc_im = hydro.sc_im.*[0 0 1 0 1 0]';
hydro.sc_ma = hydro.sc_ma.*[0 0 1 0 1 0]';
hydro.sc_ph = hydro.sc_ph.*[0 0 1 0 1 0]';
hydro.ex_re = hydro.ex_re.*[0 0 1 0 1 0]';
hydro.ex_im = hydro.ex_im.*[0 0 1 0 1 0]';
hydro.ex_ma = hydro.ex_ma.*[0 0 1 0 1 0]';
hydro.ex_ph = hydro.ex_ph.*[0 0 1 0 1 0]';

hydro = radiationIRF(hydro,[],[],[],[],[]);
hydro = radiationIRFSS(hydro,[],[]);
hydro = excitationIRF(hydro,[],[],[],[],[]);

writeBEMIOH5(hydro)

%% Plot hydro data
plotBEMIO(hydro)