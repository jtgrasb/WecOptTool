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

% plot hinge motions and relate to heave
figure()
plot(output.ptos(1).time,output.ptos(1).position(:,5)*180/pi)
hold on
plot(output.ptos(2).time,output.ptos(2).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('pto rotation (deg)')
legend('pto 1','pto 2')

bodyLocsX = [1 3];
armLength = -pto(2).location(1);

theoreticalHeave1 = bodyLocsX(1)*sin(output.ptos(1).position(:,5));
theoreticalHeave2 = armLength*sin(output.ptos(1).position(:,5)) + (bodyLocsX(2)-armLength)*sin(output.ptos(1).position(:,5) + output.ptos(2).position(:,5));

figure()
plot(output.bodies(1).time,output.bodies(1).position(:,3))
hold on
plot(output.ptos(2).time,output.bodies(2).position(:,3))
plot(output.ptos(2).time,theoreticalHeave1,'--')
plot(output.ptos(2).time,theoreticalHeave2,'--')
xlabel('time (s)')
ylabel('position (m)')
legend('body 1 heave','body 2 heave','body 1 theor','body 2 theor')