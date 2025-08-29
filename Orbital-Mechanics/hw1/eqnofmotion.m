function stateder = eqnofmotion(t, state)
% Lauren, 8/25
% the original equation is a 3D vector second order ODE
% need to translate into a system of 6 first order ODES

    % position
    posVec = state(1:3);  % [x; y; z]
    
    % r = distance from center of earth to the satellite
    % r = rm - rM (BMW 1-20)
    r = norm(posVec);
 
    % velocity
    velVec = state(4:6);  % [vx; vy; vz]
    mu = 3.986e5;  % km^3/s^2 (bc earth is the central mass
    
    % state = [xd;yd;zd;xdd;ydd;zdd]
    % aka state = [velocity; accel] where accel = (-mu/(mag(r)^3))*rvec
    stateder = [velVec; -mu/r^3 * posVec];
end