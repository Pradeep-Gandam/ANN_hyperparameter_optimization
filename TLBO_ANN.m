% Initialize TLBO parameters
num_vars = 3; % Number of decision variables
var_size = [1 num_vars]; % Size of decision variable vector
lb = [1 3 0.01]; % Lower bound of decision variables
ub = [5 50 0.9]; % Upper bound of decision variables
max_iter = 100; % Maximum number of iterations
n_pop = 30; % Population size

% Initialize TLBO population with integer values for x1 and x2
pop = zeros(n_pop, num_vars);
pop(:,1) = round(lb(1) + rand(1,n_pop) * (ub(1) - lb(1)));
pop(:,2) = round(lb(2) + rand(1,n_pop) * (ub(2) - lb(2)));
pop(:,3) = lb(3) + rand(1,n_pop) * (ub(3) - lb(3));

% Initialize variables to store best solution and cost
best_sol = [];
best_cost = inf;
best_iter = 0;

% Initialize variables to store convergence data
convergence_data = zeros(max_iter, 1);

% Main TLBO loop
for i = 1:max_iter
    % Evaluate population
    cost = zeros(n_pop, 1);
    for j = 1:n_pop
        % Call your cost function here with the current decision variable values
        X = pop(j,:);
        cost(j) = ann_cost(X);
    end
    
    % Find best solution in the population
    [local_best_cost, best_idx] = min(cost);
    local_best_sol = pop(best_idx,:);
    
    % Update global best solution
    if local_best_cost < best_cost
        best_cost = local_best_cost;
        best_sol = local_best_sol;
        best_iter = i;
    end
    
    % Calculate mean solution (centroid)
    mean_sol = mean(pop);
    
    % Generate new solutions
    for j = 1:n_pop
        % Choose a random solution from the population
        rand_idx1 = randi([1 n_pop]);
        rand_idx2 = randi([1 n_pop]);
        
        % Generate a new solution by learning
        diff = pop(rand_idx1,:) - pop(rand_idx2,:);
        new_sol = pop(j,:) + rand(1,num_vars) .* diff + rand(1,num_vars) .* (best_sol - mean_sol);
        
        % Apply boundary constraints
        new_sol = max(new_sol, lb);
        new_sol = min(new_sol, ub);
        
     
        % Evaluate new solution
        new_cost = ann_cost(new_sol);
        
        % Replace worst solution in population with new solution
        [worst_cost, worst_idx] = max(cost);
        if new_cost < cost(worst_idx)
            pop(worst_idx,:) = new_sol;
            cost(worst_idx) = new_cost;
        end
    end
    
    % Store convergence data
    convergence_data(i) = best_cost;
    
    disp(['Iteration ' num2str(i) ': Best Cost = ' num2str(best_cost) ', Best X1 = ' num2str(round(best_sol(1))) ', Best X2 = ' num2str(round(best_sol(2))) ', Best X3 = ' num2str(best_sol(3))]);
end
% Display best solution found
disp(['Best solution found: x1 = ' num2str(best_sol(1)) ', x2 = ' num2str(best_sol(2)) ', x3 = ' num2str(best_sol(3))]);
disp(['Best cost = ' num2str(best_cost) ' at iteration ' num2str(best_iter)]);

% Plot convergence data
figure;
plot(1:max_iter, convergence_data, 'LineWidth', 2.5);
xlabel('Iteration');
ylabel('Best Cost');
title('Convergence Plot'); 
