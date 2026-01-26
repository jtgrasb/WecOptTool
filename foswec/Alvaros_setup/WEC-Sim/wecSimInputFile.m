%% Simulation Data
simu = simulationClass();               % Initialize Simulation Class
simu.simMechanicsFile = 'flap.slx';      % Specify Simulink Model File
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.explorer = 'on';                   % Turn SimMechanics Explorer (on/off)
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 20;                    % Wave Ramp Time [s]
simu.endTime = 100;                     % Simulation End Time [s]
simu.solver = 'ode45';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.01; 							% Simulation time-step [s]
simu.domainSize = 50;
simu.rho = 1025;
simu.b2b = 0;

%% Wave Information 
% % noWaveCIC, no waves with radiation CIC  
% waves = waveClass('noWave');       % Initialize Wave Class and Specify Type  
% waves.period = 8;

% % Regular Waves  
waves = waveClass('regular');           % Initialize Wave Class and Specify Type                                 
waves.height = 0.01;                     % Wave Height [m]
waves.period = 8;                       % Wave Period [s]
waves.waterDepth = 0.5;

%% Body Data
% platform
% body(1) = bodyClass('hydroData/flap.h5');      
%     % Create the body(1) Variable, Set Location of Hydrodynamic Data File 
%     % and Body Number Within this File.   
% body(1).geometryFile = 'geometry/platform.stl';    % Location of Geomtry File
% body(1).mass = 350; %'equilibrium';                   
%     % Body Mass. The 'equilibrium' Option Sets it to the Displaced Water 
%     % Weight.
% body(1).inertia = [40 30 40];  % Moment of Inertia [kg*m^2]
% body(1).viz.opacity = 0.3;

% body(1) = bodyClass('hydroData/foswec.h5');
% body(1).geometryFile = 'geometry/platform.stl';
% body(1).viz.color = [1 1 1];
% body(1).viz.opacity = 0.25;
% body(1).mass = 343;%165.5;                      %[kg]  from Exp
% body(1).inertia = [37.88 29.63 53.61]; %[kg-m^2] from Exp

% flap 1
body(1) = bodyClass('hydroData/flap.h5'); 
body(1).geometryFile = 'geometry/flap.stl'; 
body(1).mass = 0.5*(.05*.76*.56)*1000; % mass of flaps is half of displaced mass to add buoyancy
body(1).inertia = [1.2 1.19 1.2];

% body(2) = bodyClass('hydroData/foswec.h5');
% body(2).geometryFile = 'geometry/flap.stl';
% body(2).mass = 0.4*(.05*.76*.56)*1000;                       %[kg] from Exp
% body(2).inertia = [1.42 1.19 1.99];    %[kg-m^2] from Exp

% flap 2
% body(3) = bodyClass('hydroData/foswec.h5'); 
% body(3).geometryFile = 'geometry/flap.stl'; 
% body(3).mass = 0.5*(.05*.76*.56)*1000;                   
% body(3).inertia = [1.2 1.19 1.2];

% body(3) = bodyClass('hydroData/foswec.h5');
% body(3).geometryFile = 'geometry/flap.stl';
% body(3).mass = 23.14/2;                       %[kg] from Exp
% body(3).inertia = [1.42 1.19 1.99];    %[kg-m^2] from Exp

%% PTO and Constraint Parameters
% Floating (3DOF) Joint
% constraint(1) = constraintClass('Constraint1'); % Initialize Constraint Class for Constraint1
% constraint(1).location = [0 0 -.54];               % Constraint Location [m]

% Translational PTO
pto(1) = ptoClass('PTO1');                      % Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 0;                             % PTO Damping [N/(m/s)]
pto(1).location = [0 0 -0.5];                      % PTO Location [m]

% pto(2) = ptoClass('PTO2');                      % Initialize PTO Class for PTO1
% pto(2).stiffness = 0;                           % PTO Stiffness [N/m]
% pto(2).damping = 0;                       % PTO Damping [N/(m/s)]
% pto(2).location = [1.44/2 0 -0.5];                      % PTO Location [m]

%% mooring to keep equilibrium
% mooring(1) = mooringClass('mooring');               % Initialize mooringClass
% mooring(1).location = [0, 0, -0.54];
% mooring(1).matrix.stiffness = diag([3*1e5 0 6e4 0 5e4 0]);
% mooring(1).matrix.damping = diag([3*1e4 0 5e4 0 5e5 0]);
% mooring(1).matrix.preTension(3) = -9.81*2*(1-0.5)*(.05*.76*.46)*1000 + 9.81*(350-237.25); % compensates for buoyancy from flaps

% 2 Moorings (one on either end)
% mooring(1) = mooringClass('mooring');               % Initialize mooringClass
% mooring(1).location = [-1.44/2, 0, -0.54];
% mooring(1).matrix.stiffness = diag([1e3 0 8e4 0 0 0]);
% mooring(1).matrix.damping = diag([1e3 0 6e4 0 0 0]);
% mooring(1).matrix.preTension(3) = -9.81*1000*((.05*.76*.46) - 0.4*(.05*.76*.56));

% mooring(2) = mooringClass('mooring');               % Initialize mooringClass
% mooring(2).location = [1.44/2, 0, -0.54];
% mooring(2).matrix.stiffness = diag([1e3 0 8e4 0 0 0]);
% mooring(2).matrix.damping = diag([1e3 0 6e4 0 0 0]);
% mooring(2).matrix.preTension(3) = -9.81*1000*((.05*.76*.46) - 0.4*(.05*.76*.56));