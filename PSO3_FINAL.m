clc;
clear all;
 
% Set PSO Parameters
num_particles = 100;     % number of particles
num_iterations = 100;    % maximum number of iterations
inertia_weight = 1.4;   % inertia weight
cognitive_factor = 1.8; % cognitive factor
social_factor = 1.8;    % social factor
 
% Set decision variable bounds
lb = [1, 3, 0.01]; % lower bounds
ub = [5, 50, 0.9];  % upper bounds
 
% Initialize particle positions and velocities
particles = zeros(num_particles, 3);
velocities = zeros(num_particles, 3);
for i=1:num_particles
    % Generate random integers for X1 and X2 within bounds
    particles(i,1) = round(rand*(ub(1)-lb(1)) + lb(1));
    particles(i,2) = round(rand*(ub(2)-lb(2)) + lb(2));
    % Generate random value for X3 within bounds
    particles(i,3) = lb(3) + (ub(3)-lb(3)).*rand(1);
    % Set initial velocities
    velocities(i,:) = -1 + 2.*rand(1,3);
end
 
% Initialize global best
global_best_cost = inf;
global_best_particle = zeros(1,3);
 
% Store iteration results in matrix
iter_results = zeros(num_iterations, 4);
 
% Begin PSO
for iter=1:num_iterations
    % Evaluate particle costs
    costs = zeros(1,num_particles);
    for j=1:num_particles
        costs(j) = ann_cost(particles(j,:));
    end
    
    % Update global best
    [min_cost, min_index] = min(costs);
    if min_cost < global_best_cost
        global_best_cost = min_cost;
        global_best_particle = particles(min_index,:);
        best_iter = iter;
    end
    
    % Update particle velocities and positions
    for j=1:num_particles
        % Update velocity
        velocities(j,:) = inertia_weight*velocities(j,:) + ...
            cognitive_factor*rand(1,3).*(particles(j,:) - particles(j,:)) + ...
            social_factor*rand(1,3).*(global_best_particle - particles(j,:));
        
        % Update position
        particles(j,:) = particles(j,:) + velocities(j,:);
        
        % Enforce bounds and integer constraints
        particles(j,1) = max(particles(j,1), lb(1));
        particles(j,1) = min(particles(j,1), ub(1));
        particles(j,2) = max(particles(j,2), lb(2));
        particles(j,2) = min(particles(j,2), ub(2));
        particles(j,1:2) = round(particles(j,1:2));
        particles(j,3) = max(particles(j,3), lb(3));
        particles(j,3) = min(particles(j,3), ub(3));
    end
    
    % Store iteration results in matrix
    iter_results(iter,:) = [global_best_particle, global_best_cost];
    
    % Display iteration info with best values of X1, X2, and X3
    disp(['Iteration ' num2str(iter) ': Best cost = ' num2str(global_best_cost) ', Best X1 = ' num2str(round(global_best_particle(1))) ', Best X2 = ' num2str(round(global_best_particle(2))) ', Best X3 = ' num2str(global_best_particle(3))]);

end

% Find final best result among all iterations
[min_cost, min_idx] = min(iter_results(:,end));
final_best_particle = iter_results(min_idx,1:end-1);

% Display final best result separately
disp(['Final best cost = ' num2str(min_cost) ', Best X1 = ' num2str(round(final_best_particle(1))) ', Best X2 = ' num2str(round(final_best_particle(2))) ', Best X3 = ' num2str(final_best_particle(3)) ',at iteration =' num2str(best_iter)]);

% Generate convergence plot
figure(1)
plot(iter_results(:,4))
title('Convergence Plot')
xlabel('Iteration')
ylabel('Best Cost')
