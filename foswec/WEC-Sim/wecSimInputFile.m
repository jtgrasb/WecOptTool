%% Simulation Data
simu = simulationClass();               % Initialize Simulation Class
simu.simMechanicsFile = 'foswec.slx';      % Specify Simulink Model File
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.explorer = 'on';                   % Turn SimMechanics Explorer (on/off)
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 20;                    % Wave Ramp Time [s]
simu.endTime = 100;                     % Simulation End Time [s]
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.001; 							% Simulation time-step [s]
simu.domainSize = 50;

%% Wave Information 
% % noWaveCIC, no waves with radiation CIC  
% waves = waveClass('noWave');       % Initialize Wave Class and Specify Type  
% waves.period = 8;

% % Regular Waves  
waves = waveClass('regular');           % Initialize Wave Class and Specify Type                                 
waves.height = 0.01;                     % Wave Height [m]
waves.period = 8;                       % Wave Period [s]

%% Body Data
% platform
body(1) = bodyClass('hydroData/foswec.h5');      
    % Create the body(1) Variable, Set Location of Hydrodynamic Data File 
    % and Body Number Within this File.   
body(1).geometryFile = 'geometry/platform.stl';    % Location of Geomtry File
body(1).mass = 'equilibrium';                   
    % Body Mass. The 'equilibrium' Option Sets it to the Displaced Water 
    % Weight.
body(1).inertia = [0 (1/12)*(1.44*0.76*0.05)*1000*(1.44^2 + 0.05^2) 0];  % Moment of Inertia [kg*m^2]
body(1).viz.opacity = 0.3;

% flap 1
body(2) = bodyClass('hydroData/foswec.h5'); 
body(2).geometryFile = 'geometry/flap.stl'; 
body(2).mass = 0.4*(.05*.76*.56)*1000; % mass of flaps is half of displaced mass to add buoyancy
body(2).inertia = [0 (1/12)*0.4*(.05*.76*.56)*1000*(0.05^2 + 0.56^2) 0];

% flap 2
body(3) = bodyClass('hydroData/foswec.h5'); 
body(3).geometryFile = 'geometry/flap.stl'; 
body(3).mass = 0.4*(.05*.76*.56)*1000;                   
body(3).inertia = [0 (1/12)*0.4*(0.05*0.76*0.56)*1000*(0.05^2 + 0.56^2) 0];

%% PTO and Constraint Parameters
% Floating (3DOF) Joint
constraint(1) = constraintClass('Constraint1'); % Initialize Constraint Class for Constraint1
constraint(1).location = [0 0 -0.46];               % Constraint Location [m]

% Translational PTO
pto(1) = ptoClass('PTO1');                      % Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 0;                             % PTO Damping [N/(m/s)]
pto(1).location = [-1.44/2+.05/2 0 -0.46];                      % PTO Location [m]

pto(2) = ptoClass('PTO2');                      % Initialize PTO Class for PTO1
pto(2).stiffness = 0;                           % PTO Stiffness [N/m]
pto(2).damping = 0;                       % PTO Damping [N/(m/s)]
pto(2).location = [1.44/2-.05/2 0 -0.46];                      % PTO Location [m]

%% mooring to keep equilibrium
mooring(1) = mooringClass('mooring');               % Initialize mooringClass
mooring(1).location = [0, 0, -0.54];
mooring(1).matrix.stiffness = diag([1e4 0 4e4 0 1e3 0]);
mooring(1).matrix.damping = diag([1e4 0 4e4 0 1e3 0]);
mooring(1).matrix.preTension(3) = 2*-9.81*1000*((.05*.76*.46) - 0.4*(.05*.76*.56));
