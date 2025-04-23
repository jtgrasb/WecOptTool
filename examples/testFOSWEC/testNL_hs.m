clear all
close all
clc

tr = stlread('flap.stl');
vertex = tr.Points;
% Check mesh size
face = tr.ConnectivityList;
norm = faceNormal(tr);
% Function to calculate the area of a triangle
v1 = vertex(face(:,3),:)-vertex(face(:,1),:);
v2 = vertex(face(:,2),:)-vertex(face(:,1),:);
av_tmp =  1/2.*(cross(v1,v2));
area_mag = sqrt(av_tmp(:,1).^2 + av_tmp(:,2).^2 + av_tmp(:,3).^2);
area = area_mag;
% Method to caculate the center coordinate of a triangle
c = zeros(length(face),3);
c(:,1) = (vertex(face(:,1),1)+vertex(face(:,2),1)+vertex(face(:,3),1))./3;
c(:,2) = (vertex(face(:,1),2)+vertex(face(:,2),2)+vertex(face(:,3),2))./3;
c(:,3) = (vertex(face(:,1),3)+vertex(face(:,2),3)+vertex(face(:,3),3))./3;
center = c;

% need to make hinge = cg to get accurate result

axis = [0 1 0];
angleVec = linspace(-1, 1, 21);
cgOrig = [0 0 -0.53+0.046];
relCoord = cgOrig - [0 0 -0.53-.046];

for ii = 1:length(angleVec)

    rotMat = axisAngle2RotMat(axis,angleVec(ii))*eye(3);
    rotatedRelCoord = relCoord*(rotMat');
    linDisp = rotatedRelCoord - relCoord;
    
    x = [0; 0; 0; 0; 0; 0]; % x is displacement in 6 degrees of freedom
    x(1:3) = linDisp;
    x(4:6) = [0, angleVec(ii), 0];
    elv = 0; % wave elevation (instantaneous)
    rho = 1000;
    g = 9.81;
    cg = [0; 0; -0.53+0.17];
    mass = 0; % set to 0 for now
    
    [f,p]  = nonLinearBuoyancy(x,elv,center,norm,area,rho,g,cg,mass);
    forceHS(ii,:) = f;
    stiffness(ii) = f(5)/angleVec(ii);
end

figure()
plot(angleVec,forceHS(:,5)) % this is the force at the center of gravity
xlabel('angle (rad)')
ylabel('torque (Nm)')

figure()
plot(angleVec,stiffness)
xlabel('angle (rad)')
ylabel('stiffness (Nms/rad)')

