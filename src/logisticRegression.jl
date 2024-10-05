using LinearAlgebra
using Plots

# Sigmoid function
function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp.(-z))
end

# Hypothesis function
function hypothesis(X, theta)
    return sigmoid(X * theta)
end

# Cost function
function compute_cost(X, y, theta)
    m = length(y)
    h = hypothesis(X, theta)
    return -(1 / m) * sum(y .* log.(h) + (1 .- y) .* log.(1 .- h))
end

# Gradient Descent
function gradient_descent(X, y, theta, alpha, num_iters)
    m = length(y)
    cost_history = zeros(num_iters)

    for i in 1:num_iters
        h = hypothesis(X, theta)
        theta = theta - (alpha / m) * (X' * (h - y))
        cost_history[i] = compute_cost(X, y, theta)
    end

    return theta, cost_history
end

# Visualizing the cost function history
function plot_cost_history(cost_history)
    p = plot(1:length(cost_history), cost_history, title="Cost Function History",
        xlabel="Iteration", ylabel="Cost", linewidth=2, legend=false)
    display(p)  # Make sure the plot is displayed
end

# Visualizing the data
function plot_features(X, y, theta)
    p = scatter(X[:, 2], X[:, 3], group=y, title="Logistic Regression: Feature Plot",
        xlabel="Feature 1", ylabel="Feature 2", label=["Class 0" "Class 1"], legend=:top)

    # Plot the decision boundary
    x_vals = minimum(X[:, 2]):0.1:maximum(X[:, 2])
    y_vals = (-theta[1] .- theta[2] .* x_vals) ./ theta[3]  # Decision boundary
    plot!(x_vals, y_vals, label="Decision Boundary", lw=2)
    display(p)  # Make sure the plot is displayed
end

# Example usage:
X = [1.0 2.0 3.0; 1.0 3.0 4.0; 1.0 5.0 6.0; 1.0 7.0 8.0] # Add intercept term (ones in first column)
y = [0.0, 0.0, 1.0, 1.0] # Labels (0 or 1)

# Initialize parameters
theta = zeros(size(X, 2)) # Weights initialized to zero
alpha = 0.01              # Learning rate
num_iters = 1000          # Number of iterations

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)

# Print the final weights
println("Theta after gradient descent: $theta")

# Visualize the cost function history
plot_cost_history(cost_history)

# Visualize the feature data and the decision boundary
plot_features(X, y, theta)

println("Press Enter to exit...")
readline()
