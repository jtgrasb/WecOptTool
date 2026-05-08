%Example of user input MATLAB file for post processing
close all

figure()
plot(output.bodies(1).time, output.bodies(1).position(:,1))
hold on
plot(output.bodies(1).time, output.bodies(1).position(:,3))
plot(output.bodies(1).time, output.bodies(1).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('platform response (m or deg)')
legend('surge','heave','pitch')

figure()
plot(output.ptos(1).time, output.ptos(1).position(:,5)*180/pi)
hold on
plot(output.ptos(1).time, output.ptos(2).position(:,5)*180/pi)
xlabel('time (s)')
ylabel('pto rotation (deg)')
legend('flap 1','flap2')

% save output data
time = output.bodies(1).time;
platformResponse = output.bodies(1).position;
flap1Response = output.bodies(2).position;
flap2Response = output.bodies(3).position;
pto1Response = output.ptos(1).position;
pto2Response = output.ptos(2).position;
save WSoutput.mat time platformResponse flap1Response flap2Response pto1Response pto2Response


