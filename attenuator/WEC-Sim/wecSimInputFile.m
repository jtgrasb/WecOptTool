%% Simulation Data
simu = simulationClass();               % Initialize Simulation Class
simu.simMechanicsFile = 'attenuator.slx';      % Specify Simulink Model File
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.explorer = 'on';                   % Turn SimMechanics Explorer (on/off)
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 20;                    % Wave Ramp Time [s]
simu.endTime = 100;                     % Simulation End Time [s]
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.01; 							% Simulation time-step [s]

%% Wave Information 
% % noWaveCIC, no waves with radiation CIC  
% waves = waveClass('noWaveCIC');       % Initialize Wave Class and Specify Type  

% % Regular Waves  
waves = waveClass('regular');           % Initialize Wave Class and Specify Type                                 
waves.height = .1;                     % Wave Height [m]
waves.period = 8;                       % Wave Period [s]


%% Body Data
% Float
body(1) = bodyClass('hydroData/attenuator.h5');      
    % Create the body(1) Variable, Set Location of Hydrodynamic Data File 
    % and Body Number Within this File.   
body(1).geometryFile = 'geometry/cyl_0_5.stl';    % Location of Geomtry File
body(1).mass = 'equilibrium';                   
    % Body Mass. The 'equilibrium' Option Sets it to the Displaced Water 
    % Weight.
body(1).inertia = [10 10 10];  % Moment of Inertia [kg*m^2]     

% Spar/Plate
body(2) = bodyClass('hydroData/attenuator.h5'); 
body(2).geometryFile = 'geometry/cyl_0_5.stl'; 
body(2).mass = 'equilibrium';                   
body(2).inertia = [10 10 10];

%% PTO and Constraint Parameters
% Floating (3DOF) Joint

% Translational PTO
pto(1) = ptoClass('PTO1');                      % Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 1e3;                       % PTO Damping [N/(m/s)]
pto(1).location = [0 0 0];                      % PTO Location [m]

pto(2) = ptoClass('PTO2');                      % Initialize PTO Class for PTO1
pto(2).stiffness = 0;                           % PTO Stiffness [N/m]
pto(2).damping = 1e3;                       % PTO Damping [N/(m/s)]
pto(2).location = [0 0 0];                      % PTO Location [m]
