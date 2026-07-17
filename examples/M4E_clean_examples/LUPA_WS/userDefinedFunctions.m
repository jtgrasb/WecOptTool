%Example of user input MATLAB file for post processing
close all

figure()
plot(output.bodies(1).time, output.bodies(1).position(:,1)-body(1).centerGravity(1))
hold on
plot(output.bodies(1).time, output.bodies(1).position(:,3)-body(1).centerGravity(3))
plot(output.bodies(1).time, output.bodies(1).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('spar response (m or deg)')
legend('surge','heave','pitch')

figure()
plot(output.bodies(1).time, output.bodies(2).position(:,1)-body(2).centerGravity(1))
hold on
plot(output.bodies(1).time, output.bodies(2).position(:,3)-body(2).centerGravity(3))
plot(output.bodies(1).time, output.bodies(2).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('float response (m or deg)')
legend('surge','heave','pitch')

% save output data
time = output.bodies(1).time;
waveElev = waves.waveAmpTime(:,2);
sparResponse = output.bodies(1).position;
floatResponse = output.bodies(2).position;
ptoResponse = output.ptos(1).position;
save WSoutput.mat time waveElev sparResponse floatResponse ptoResponse


