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
plot(output.ptos.time,output.ptos.position(:,5)*180/pi)
xlabel('time (s)')
ylabel('PTO rotation (deg)')

figure()
plot(output.ptos.time,output.ptos.powerInternalMechanics(:,5))
xlabel('time (s)')
ylabel('Mech power (W)')
 

%Save waves and response as video
% output.saveViz(simu,body,waves,...
%     'timesPerFrame',5,'axisLimits',[-150 150 -150 150 -50 20],...
%     'startEndTime',[100 125]);
