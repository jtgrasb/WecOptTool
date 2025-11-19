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

%Plot heave response for body 2
output.plotResponse(2,3);

% plot platform 1 motion
figure()
plot(output.bodies(1).time,output.bodies(1).position(:,1)-output.bodies(1).centerGravity(1))
hold on
plot(output.bodies(1).time,output.bodies(1).position(:,3)-output.bodies(1).centerGravity(3))
plot(output.bodies(1).time,output.bodies(1).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('platform response (m or deg)')
legend('surge','heave','pitch')
% xlim([50,50+waves.period])
grid on

% plot hinge motions 
figure()
plot(output.ptos(1).time,output.ptos(1).position(:,5)*180/pi)
hold on
plot(output.ptos(2).time,output.ptos(2).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('pto rotation (deg)')
legend('pto 1','pto 2')
% xlim([50,50+waves.period])
grid on
