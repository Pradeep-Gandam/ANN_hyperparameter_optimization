function [xopt, fopt, output] = GA_Final1()
% Define the bounds for the hyperparameters
lb = [1, 3, 0.01]; % Lower bounds for x1, x2, x3
ub = [5, 50, 0.9]; % Upper bounds for x1, x2, x3

% Define the GA options
options = optimoptions('ga');
options.Display = 'iter';
options.PlotFcn = {@gaplotbestf,};
options.MaxGenerations = 100; % Set the maximum number of generations
options.PopulationSize = 100; % Set the population size

% Define the fitness function
fitnessfcn = @(x) ann_cost(x);

% Run the GA optimization
[xopt, fopt, exitflag, output] = ga(fitnessfcn, 3, [], [], [], [], lb, ub, [], options);

% Extract the best solution
x1opt = round(xopt(1));
x2opt = round(xopt(2));
x3opt = xopt(3);

% Print the results
fprintf('Best cost: %.4f\n', fopt);
fprintf('Best x1: %d\n', x1opt);
fprintf('Best x2: %d\n', x2opt);
fprintf('Best x3: %.4f\n', x3opt);
fprintf('Best generation: %d\n', output.generations);
end
