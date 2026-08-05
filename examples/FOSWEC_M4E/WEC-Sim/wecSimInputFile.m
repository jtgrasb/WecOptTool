%% Simulation Data
simu = simulationClass();               % Initialize Simulation Class
simu.simMechanicsFile = 'foswec.slx';      % Specify Simulink Model File
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.explorer = 'on';                   % Turn SimMechanics Explorer (on/off)
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 100;                    % Wave Ramp Time [s]
simu.endTime = 300;                     % Simulation End Time [s]
% simu.rampTime = 5*2.61;
% simu.endTime = 20*2.61;
% simu.dt = 2.61/200;
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.002; 							% Simulation time-step [s]
simu.domainSize = 50;
simu.rho = 1025;
simu.b2b = 1;
simu.mcrMatFile = 'mcrCases.mat';

%% Wave Information 
% noWaveCIC, no waves with radiation CIC  
% waves = waveClass('noWave');       % Initialize Wave Class and Specify Type  
% waves.period = 8;

% Regular Waves  
waves = waveClass('regular');           % Initialize Wave Class and Specify Type                                 
waves.height = 0.136;                     % Wave Height [m]
waves.period = 2.61;                       % Wave Period [s]

waves = waveClass('regular');           % Initialize Wave Class and Specify Type                                 
waves.height = 2*0.136;                     % Wave Height [m]
waves.period = 2.61;                       % Wave Period [s]

%% Body Data

% platform
body(1) = bodyClass('hydroData/foswec_short_period.h5');      
body(1).geometryFile = 'geometry/platform.stl';    % Location of Geomtry File
body(1).mass = 189.8; %'equilibrium';                   
body(1).inertia = [30 30 30];  % Moment of Inertia [kg*m^2]
body(1).viz.opacity = 0.3;
% body(1).adjMassFactor = 1.5;

% flap 1
body(2) = bodyClass('hydroData/foswec_short_period.h5'); 
body(2).geometryFile = 'geometry/flap.stl'; 
body(2).mass = 23.1; % mass of flaps is less than displaced mass to add buoyancy
% body(2).mass = 12.1; 
body(2).inertia = [1.19 1.19 1.19];
% body(2).adjMassFactor = 1.5;
body(2).linearDamping = zeros(6);
% body(2).linearDamping(1,1) = 500;
% body(2).linearDamping(5,1) = 10;  
body(2).linearDamping(5,5) = 10; 

% flap 2
body(3) = bodyClass('hydroData/foswec_short_period.h5'); 
body(3).geometryFile = 'geometry/flap.stl'; 
body(3).mass = 23.1; 
% body(3).mass = 12.1; 
body(3).inertia = [1.19 1.19 1.19];
% body(3).adjMassFactor = 1.5;
body(3).linearDamping = zeros(6);
% body(3).linearDamping(1,1) = 500;
% body(3).linearDamping(5,1) = 10;  
body(3).linearDamping(5,5) = 10;  


%% PTO and Constraint Parameters
% Floating (3DOF) Joint
constraint(1) = constraintClass('Constraint1'); % Initialize Constraint Class for Constraint1
constraint(1).location = [0 0 -.8];               % Constraint Location [m]

% Translational constraints for surge and heave motion
% constraint(1) = constraintClass('Constraint1'); % Initialize Constraint Class for Constraint1
% constraint(1).location = [0 0 -0.8];               % Constraint Location [m]
% 
% constraint(2) = constraintClass('Constraint2'); % Initialize Constraint Class for Constraint1
% constraint(2).location = [0 0 -0.8];               % Constraint Location [m]
% constraint(2).orientation.z = [1, 0, 0];
% constraint(2).orientation.y = [0, 1, 0];

% Translational PTO
pto(1) = ptoClass('PTO1');                      % Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 0;                             % PTO Damping [N/(m/s)]
pto(1).location = [-0.72, 0, -0.59];                      % PTO Location [m]

pto(2) = ptoClass('PTO2');                      % Initialize PTO Class for PTO1
pto(2).stiffness = 0;                           % PTO Stiffness [N/m]
pto(2).damping = 0;                       % PTO Damping [N/(m/s)]
pto(2).location = [0.72, 0, -0.59];                       % PTO Location [m]

%% mooring to keep equilibrium
totalMass = body(1).mass + body(2).mass + body(3).mass;
displacedVolumePlatform = 0.195603434649359;
displacedVolumeFlaps = 2*0.030856000000000;
displacedMass = simu.rho*(displacedVolumePlatform + displacedVolumeFlaps);

mooring(1) = mooringClass('mooring');               % Initialize mooringClass
mooring(1).location = [0, 0, -0.8];
mooring(1).matrix.preTension(3) = -9.81*(displacedMass - totalMass); % -9.81*(displacedMass - totalMass)
mooring(1).matrix.stiffness = diag([8e3 0 2e5 0 2e5 0]);
mooring(1).matrix.damping = diag([8e2 0 1e4 0 1e4 0]);

%% added from foswec v2

% % Body 4: Mooring Non-hydro Body (Mooring Line 1)
% body(4) = bodyClass('');                % Initialize bodyClass without an *.h5 file
% body(4).geometryFile = 'geometry/squares.stl';    % Geometry File
% body(4).nonHydro = 1;                     % Turn non-hydro body on
% body(4).name = 'line_1';                  % Specify body name
% body(4).mass = 0.01;                     % Specify Mass  
% body(4).inertia = [0 0 0];         % Specify MOI  
% body(4).centerGravity = [-0.72 0 -1.2];                % Specify Cg  
% body(4).volume = 0;                    % Specify Displaced Volume  
% body(4).centerBuoyancy = [0,0,0];
% 
% % Body 4: Mooring Non-hydro Body (Mooring Line 2)
% body(5) = bodyClass('');                % Initialize bodyClass without an *.h5 file
% body(5).geometryFile = 'geometry/squares.stl';    % Geometry File
% body(5).nonHydro = 1;                     % Turn non-hydro body on
% body(5).name = 'line_2';                  % Specify body name
% body(5).mass = 0.01;                     % Specify Mass  
% body(5).inertia = [0 0 0];         % Specify MOI  
% body(5).centerGravity = [0.72 0 -1.2];                % Specify Cg  
% body(5).volume = 0;                    % Specify Displaced Volume  
% body(5).centerBuoyancy = [0,0,0];
% 
% %% PTO 3: Rotational PTO (Connection 1)
% pto(3) = ptoClass('connection_1');
% pto(3).location = [-0.72 0 -1.05];
% % apply Mooring stiffness 
% pto(3).stiffness = (1e4)/10;     %based on stiffness from previous model
% pto(3).damping = 0;
% 
% %% PTO 4:  Rotational PTO (Anchor 1)
% pto(4) = ptoClass('anchor_1');
% pto(4).location = [-0.72 0 -1.35];
% pto(4).stiffness = 0;
% pto(4).damping = 0;
% 
% %% PTO 5: Rotational PTO (Connection 2)
% pto(5) = ptoClass('connection_2');
% pto(5).location = [0.72 0 -1.05];
% % apply Mooring stiffness 
% pto(5).stiffness = (1e4)/10;     %based on stiffness from previous model
% pto(5).damping = 0;
% 
% %% PTO 6:  Rotational PTO (Anchor 2)
% pto(6) = ptoClass('anchor_2');
% pto(6).location = [0.72 0 -1.35];
% pto(6).stiffness = 0;
% pto(6).damping = 0;