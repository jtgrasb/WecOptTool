%Example of user input MATLAB file for post processing
close all

%Plot waves
waves.plotElevation(simu.rampTime);
try 
    waves.plotSpectrum();
catch
end

%Plot heave response for body 1
output.plotResponse(1,3);

%Plot heave forces for body 1
output.plotForces(1,3);

figure()
plot(output.ptos(1).time,output.ptos(1).powerInternalMechanics(:,5))
title('pto 1')
xlabel('time (s)')
ylabel('Mech power (W)')

figure()
plot(output.bodies(1).time, output.bodies(1).position(:,5)*180/pi)
hold on
plot(output.bodies(1).time, output.bodies(1).position(:,3) - body(1).centerGravity(3))
title('body 1')
xlabel('time (s)')
ylabel('m or deg')
legend('pitch','heave')

% theoretical heave body 2
theoreticalHeave = 3*(5)*sin(output.ptos(1).position(:,5)) + (5)*sin(output.ptos(2).position(:,5));

figure()
plot(output.bodies(2).time, output.bodies(2).position(:,5)*180/pi)
hold on
plot(output.bodies(2).time, output.bodies(2).position(:,3) - body(2).centerGravity(3))
plot(output.bodies(2).time, theoreticalHeave - body(2).centerGravity(3),'--')
title('body 2')
xlabel('time (s)')
ylabel('m or deg')
legend('pitch','heave','theoretical heave')

figure()
plot(output.ptos(1).time, output.ptos(1).position(:,5)*180/pi)
hold on
plot(output.ptos(2).time, output.ptos(2).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('deg')
legend('pto 1','pto 2')

figure()
yyaxis left
plot(output.ptos(1).time, output.ptos(1).velocity(:,5))
yyaxis right
plot(output.ptos(1).time, output.ptos(1).forceInternalMechanics(:,5))
xlabel('time (s)')
ylabel('rad/s or Nm')
legend('pto 1 vel','pto 1 force')

figure()
yyaxis left
plot(output.ptos(2).time, output.ptos(2).velocity(:,5))
yyaxis right
plot(output.ptos(2).time, output.ptos(2).forceInternalMechanics(:,5))
xlabel('time (s)')
ylabel('rad/s or Nm')
legend('pto 2 vel','pto 2 force')

figure()
plot(output.ptos(1).time,output.ptos(1).powerInternalMechanics(:,5))
title('pto 1')
xlabel('time (s)')
ylabel('Mech power (W)')

figure()
plot(output.ptos(2).time,output.ptos(2).powerInternalMechanics(:,5))
title('pto 2')
xlabel('time (s)')
ylabel('Mech power (W)')

%Save waves and response as video
% output.saveViz(simu,body,waves,...
%     'timesPerFrame',5,'axisLimits',[-150 150 -150 150 -50 20],...
%     'startEndTime',[100 125]);

%% plot forces

% What if I run BEM with only 2 points like with WOT?

% plots the forces on body 1 converted to be about pto 1
excF1 = output.bodies(1).forceExcitation(:,3)*5 + output.bodies(1).forceExcitation(:,5);
ptoF1 = output.ptos(1).forceInternalMechanics(:,5);
radF1 = (output.bodies(1).forceRadiationDamping(:,3)+output.bodies(1).forceAddedMass(:,3))*5 + (output.bodies(1).forceRadiationDamping(:,5)+output.bodies(1).forceAddedMass(:,5));
hsF1 = output.bodies(1).forceRestoring(:,3)*5 + output.bodies(1).forceRestoring(:,5);

figure()
plot(output.bodies(1).time, excF1)

figure()
plot(output.bodies(1).time, ptoF1)

figure()
plot(output.bodies(1).time, radF1)

figure()
plot(output.bodies(1).time, hsF1)

