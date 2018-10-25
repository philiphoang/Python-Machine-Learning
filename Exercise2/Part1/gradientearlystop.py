from gradientdescent import step_gradient
from mse import compute_error_for_line_given_points

def gradient_descent_runner_early_stop(points, starting_w0, starting_w1, learning_rate, num_iterations, threshold):
    w0 = starting_w0
    w1 = starting_w1
    for i in range(num_iterations):
            old_w0 = w0
            old_w1 = w1

            w0, w1 = step_gradient(w0, w1, points, learning_rate)

            old_mse = compute_error_for_line_given_points(old_w0, old_w1, points)

            mse = compute_error_for_line_given_points(w0, w1, points)
            check = old_mse - mse
            print(check)
            if (check < threshold):
                print(f'Iteration {i+1}: w0={w0:0.5f}, w1={w1:0.5f}, mse={mse:0.5f} Stopped')
                return [w0, w1, mse]
            print(f'Iteration {i+1}: w0={w0:0.5f}, w1={w1:0.5f}, mse={mse:0.5f}')
    return [w0, w1, mse]
