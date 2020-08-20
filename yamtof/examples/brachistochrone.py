from yamtof.mocp import MultiPhaseOptimalControlProblem
from casadi import cos, sin, pi, sqrt, linspace, fabs, mmax, DM

if __name__ == '__main__':
    mocp = MultiPhaseOptimalControlProblem()
    start = lambda a: mocp.start(a)
    end   = lambda a: mocp.end(a)
    phase_name = 'phaseA'
    duration = mocp.create_phase(phase_name, init=0.01, n_intervals=1)

    # Add time-dependent variables
    x            = mocp.add_trajectory(phase_name, 'x')
    y            = mocp.add_trajectory(phase_name, 'y')
    speed        = mocp.add_trajectory(phase_name, 'speed', init=1.0)
    path_angle   = mocp.add_trajectory(phase_name, 'path_angle')
    angular_rate = mocp.add_trajectory(phase_name, 'angular_rate', init=0.4)

    # Slack variables for final state soft constraints
    slack_final_x = mocp.add_variable('slack_final_x', init=10.0)
    slack_final_y = mocp.add_variable('slack_final_y', init=10.0)
    mocp.add_constraint(slack_final_x > 0)
    mocp.add_constraint(slack_final_y > 0)

    param_final_x   = mocp.add_parameter('param_final_x', init=(1.5 * pi + 1.0))
    param_initial_y = mocp.add_parameter('param_initial_y', init=1.0)

    # Dynamics: Ball rolling on a ramp with a controllable slope angle
    mocp.set_derivative(x, speed * cos(path_angle))
    mocp.set_derivative(y, speed * sin(path_angle))
    mocp.set_derivative(speed,    -sin(path_angle))
    mocp.set_derivative(path_angle,   angular_rate)

    # Hard initial constraints
    mocp.add_constraint(start(x)     == 0)
    mocp.add_constraint(start(y)     == param_initial_y)
    mocp.add_constraint(start(speed) == 0)

    # Soft final constraints
    mocp.add_constraint(end(x) < param_final_x + slack_final_x)
    mocp.add_constraint(end(x) > param_final_x - slack_final_x)
    mocp.add_constraint(end(y) <  slack_final_y)
    mocp.add_constraint(end(y) > -slack_final_y)

    # Disallow multiple rotations
    mocp.add_path_constraint(path_angle > -pi)
    mocp.add_path_constraint(path_angle < pi)

    # Positive phase duration
    mocp.add_constraint(duration > 0.01)

    # Minimize time and distance to target
    mocp.add_objective(100 * slack_final_x)
    mocp.add_objective(100 * slack_final_y)
    mocp.add_objective(duration)

    # Add mechanical energy output, to demonstrate output evaluation
    mocp.add_path_output('energy', y + 0.5*speed**2)

    ### Problem done, solve it
    mocp.solve()
    mocp.phases[phase_name].change_time_resolution(8) # Refine mesh and solve again
    mocp.solve()

    # Interpolate result and compare with analytic solution
    tau_grid = linspace(0.0,1.0,801)
    t_grid = tau_grid * mocp.get_value(duration)
    result_interpolated = mocp.phases[phase_name].interpolate(tau_grid)

    analytic_solution = dict()
    analytic_solution['x']          = t_grid - sin(t_grid)
    analytic_solution['y']          = cos(t_grid)
    analytic_solution['speed']      = sqrt(2.0 - 2.0 * cos(t_grid))
    analytic_solution['path_angle'] = (t_grid - pi)/2
    analytic_solution_duration = 1.5 * pi

    print('error duration:     ', mmax(fabs(mocp.get_value(duration) - analytic_solution_duration)))
    print('error energy:       ', mmax(fabs(DM(result_interpolated['outputs']['energy']) - 1.0)))
    for trajectory_name in analytic_solution:
        errors = result_interpolated['trajectories'][trajectory_name] - analytic_solution[trajectory_name]
        print(('error ' + trajectory_name + ':                  ')[:20], mmax(fabs(errors)))

