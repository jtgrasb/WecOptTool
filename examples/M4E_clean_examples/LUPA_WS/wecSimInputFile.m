%% Simulation Data
simu = simulationClass();               % Initialize Simulation Class
simu.simMechanicsFile = 'LUPA.slx';      % Specify Simulink Model File
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.explorer = 'on';                   % Turn SimMechanics Explorer (on/off)
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 100;                    % Wave Ramp Time [s]
simu.endTime = 300;                     % Simulation End Time [s]
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.02; 							% Simulation time-step [s]
simu.domainSize = 50;
simu.rho = 1025;
simu.b2b = 1;
simu.mcrMatFile = 'mcrCases.mat';

%% Wave Information 
wavefreq = 0.4;
T = 1/wavefreq;

% noWaveCIC, no waves with radiation CIC
% waves = waveClass('noWave');       % Initialize Wave Class and Specify Type  
% waves.period = T;

% Regular Waves  
waves = waveClass('regular');           % Initialize Wave Class and Specify Type                                 
waves.height = 0.01;                     % Wave Height [m]
waves.period = T;                       % Wave Period [s]

%% Body Data

% spar
body(1) = bodyClass('hydroData/lupa_hs.h5');      
body(1).geometryFile = 'geometry/spar.stl';    % Location of Geomtry File
body(1).mass = 175.536; %'equilibrium';                   
body(1).inertia = [253.6344, 250.4558, 12.746];  % Moment of Inertia [kg*m^2]

% float
body(2) = bodyClass('hydroData/lupa_hs.h5'); 
body(2).geometryFile = 'geometry/float.stl'; 
body(2).mass = 248.721; % mass of flaps is less than displaced mass to add buoyancy
body(2).inertia = [66.1686, 65.3344, 17.16];

%% PTO and Constraint Parameters
% Floating (3DOF) Joint
constraint(1) = constraintClass('Constraint1'); % Initialize Constraint Class for Constraint1
constraint(1).location = [0 0 -1.3];               % Constraint Location [m]

% Translational PTO
pto(1) = ptoClass('PTO1');                      % Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 0;                             % PTO Damping [N/(m/s)]
pto(1).location = [0.01, 0, 0.06];                      % PTO Location [m]

%% Add mooring to maintain equilibrium

displacedVolumeSpar = 0.202656891540766;
displacedMassSpar = simu.rho*displacedVolumeSpar;

mooring(1) = mooringClass('mooring');               % Initialize mooringClass
mooring(1).location = [0, 0, -1.3];
mooring(1).matrix.stiffness = diag([1e3 0 0 0 1e3 0]);
mooring(1).matrix.damping = diag([1e2 0 0 0 1e2 0]);
mooring(1).matrix.preTension(3) = -9.81*(displacedMassSpar - body(1).mass); % -9.81*(displacedMass - totalMass)

displacedVolumeFloat = 0.239591280759531;
displacedMassFloat = simu.rho*displacedVolumeFloat;

floatCB = [-3.883345171911409e-06;-5.790529786108348e-07;-0.007990774796600];
floatCG = [0.01, 0, 0.06];

buoyancyTorquePitch = 9.81*simu.rho*displacedVolumeFloat*(floatCG(1) - floatCB(1)); 

mooring(2) = mooringClass('mooring');               % Initialize mooringClass
mooring(2).location = [0.01, 0, 0.06];
mooring(2).matrix.preTension(3) = -9.81*(displacedMassFloat - body(2).mass); % -9.81*(displacedMass - totalMass)
mooring(2).matrix.preTension(5) = -buoyancyTorquePitch;