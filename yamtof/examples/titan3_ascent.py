# Example Problem from
# Practical Methods for Optimal Control and Estimation Using
# Nonlinear Programming SECOND EDITION -- John T. Betts
# Section 6.15 Delta III Launch Vehicle, pages 336 -- 345

# Target orbit constraints based on angular momentum and eccentricity vectors:
# Real-time optimal guidance -- Brown and Johnson, 1967
# https://doi.org/10.1109/TAC.1967.1098718

import collections
import casadi
import math
from casadi import pi
from yamtof.mocp import MultiPhaseOptimalControlProblem

def main():
    p = get_parameters()

    n_phases = 4
    phase_names = ['boost6', 'boost3', 'main', 'upper']
    phase = [None] * n_phases

    mocp = MultiPhaseOptimalControlProblem()
    start = lambda a: mocp.start(a)
    end = lambda a: mocp.end(a)

    # Create the states, controls, constraints and dynamics for each phase
    for i in range(n_phases):
        phase[i] = create_rocket_stage_phase(mocp, phase_names[i], i, p)

    # Initial state constraint
    mocp.add_constraint(start(phase[0].rx) == p.initial_position_x)
    mocp.add_constraint(start(phase[0].ry) == p.initial_position_y)
    mocp.add_constraint(start(phase[0].rz) == p.initial_position_z)
    mocp.add_constraint(start(phase[0].vx) == p.initial_velocity_x)
    mocp.add_constraint(start(phase[0].vy) == p.initial_velocity_y)
    mocp.add_constraint(start(phase[0].vz) == p.initial_velocity_z)

    # Phase linkages
    for i in range(n_phases-1):
        mocp.add_constraint(start(phase[i+1].rx) == end(phase[i].rx))
        mocp.add_constraint(start(phase[i+1].ry) == end(phase[i].ry))
        mocp.add_constraint(start(phase[i+1].rz) == end(phase[i].rz))
        mocp.add_constraint(start(phase[i+1].vx) == end(phase[i].vx))
        mocp.add_constraint(start(phase[i+1].vy) == end(phase[i].vy))
        mocp.add_constraint(start(phase[i+1].vz) == end(phase[i].vz))


    # Target orbit - soft constraints
    # h_T - cross(r_f, v_f) == 0
    # e_T + cross(h_T, v_f) + r_f / norm(r_f) == 0
    r_f = casadi.vertcat(end(phase[-1].rx), end(phase[-1].ry), end(phase[-1].rz))
    v_f = casadi.vertcat(end(phase[-1].vx), end(phase[-1].vy), end(phase[-1].vz))
    h_T = casadi.SX(p.target_angular_momentum)
    e_T = casadi.SX(p.target_eccentricity)
    defect_angular_momentum = h_T - casadi.cross(r_f, v_f)
    defect_eccentricity = e_T + casadi.cross(h_T, v_f) + r_f / casadi.norm_2(r_f)

    slack_angular_momentum = mocp.add_variable('slack_angular_momentum', init=3.0)
    slack_eccentricity = mocp.add_variable('slack_eccentricity', init=3.0)

    mocp.add_constraint(slack_angular_momentum > 0)
    mocp.add_constraint(slack_eccentricity > 0)

    mocp.add_constraint(defect_angular_momentum[0] <  slack_angular_momentum)
    mocp.add_constraint(defect_angular_momentum[0] > -slack_angular_momentum)
    mocp.add_constraint(defect_angular_momentum[1] <  slack_angular_momentum)
    mocp.add_constraint(defect_angular_momentum[1] > -slack_angular_momentum)
    mocp.add_constraint(defect_angular_momentum[2] <  slack_angular_momentum)
    mocp.add_constraint(defect_angular_momentum[2] > -slack_angular_momentum)

    mocp.add_constraint(defect_eccentricity[0] <  slack_eccentricity)
    mocp.add_constraint(defect_eccentricity[0] > -slack_eccentricity)
    mocp.add_constraint(defect_eccentricity[1] <  slack_eccentricity)
    mocp.add_constraint(defect_eccentricity[1] > -slack_eccentricity)
    mocp.add_constraint(defect_eccentricity[2] <  slack_eccentricity)
    mocp.add_constraint(defect_eccentricity[2] > -slack_eccentricity)

    mocp.add_objective(100.0 * slack_angular_momentum)
    mocp.add_objective(100.0 * slack_eccentricity)

    # Maximize final mass
    final_mass = end(phase[-1].mass)
    mocp.add_objective(-1.0 * final_mass)

    ### Problem done, solve it
    mocp.solve()
    for phase_name in mocp.phases:
        mocp.phases[phase_name].change_time_resolution(6)
    mocp.solve()

    print('solution final time = ', sum([p.duration_value for p in mocp.phases.values()]) * p.scale.time)
    print('expected final time =  924.139')

    # Interpolate resulting tajectory
    tau_grid = casadi.linspace(0.0,1.0,501)
    interpolated_results = dict()
    for phase_name in mocp.phases:
        interpolated_results[phase_name] = mocp.phases[phase_name].interpolate(tau_grid)

    # Concatenate phases into complete timeline
    durations = [e['duration'] for e in interpolated_results.values()]
    trajectories = [e['trajectories'] for e in interpolated_results.values()]
    t_offset = casadi.vertsplit(casadi.cumsum(casadi.DM([0]+durations[:-1])))
    t_grid = casadi.vertcat(*[e[1] + e[0] * tau_grid for e in zip(durations, t_offset)])
    trajectories_concatendated = dict()

    for trajectory_name in trajectories[0]:
        trajectories_concatendated[trajectory_name] = [e for i in range(len(trajectories)) for e in trajectories[i][trajectory_name]]

    # Reproduce the figures from the book example
    generate_figures(t_grid, trajectories_concatendated, p)

def create_rocket_stage_phase(mocp, phase_name, phase_index, p):
    duration = mocp.create_phase(phase_name, init=0.00001, n_intervals=2)

    # Total vessel mass
    mass  = mocp.add_trajectory(phase_name, 'mass', init=p.phase_initialMass[phase_index])

    # ECI position vector
    rx    = mocp.add_trajectory(phase_name, 'rx', init=p.initial_position_x)
    ry    = mocp.add_trajectory(phase_name, 'ry', init=p.initial_position_y)
    rz    = mocp.add_trajectory(phase_name, 'rz', init=p.initial_position_z)

    # ECI velocity vector
    vx    = mocp.add_trajectory(phase_name, 'vx', init=p.initial_velocity_x)
    vy    = mocp.add_trajectory(phase_name, 'vy', init=p.initial_velocity_y)
    vz    = mocp.add_trajectory(phase_name, 'vz', init=p.initial_velocity_z)

    # ECI steering unit vector
    ux    = mocp.add_trajectory(phase_name, 'ux', init=1.0)
    uy    = mocp.add_trajectory(phase_name, 'uy', init=0.01)
    uz    = mocp.add_trajectory(phase_name, 'uz', init=0.01)

    ## Dynamics
    vr_x = vx + p.body_rotation_speed * ry
    vr_y = vy - p.body_rotation_speed * rx
    vr_z = vz

    r_squared = rx**2 + ry**2 + rz**2 + 1e-8
    vr_squared = vr_x**2 + vr_y**2 + vr_z**2 + 1e-8
    vr = vr_squared**0.5
    r = r_squared**0.5
    r3_inv = r_squared**(-1.5)

    # Gravitational acceleration (mu=1 is omitted)
    gx = -r3_inv * rx
    gy = -r3_inv * ry
    gz = -r3_inv * rz

    # Thrust force
    Tx = p.phase_thrust[phase_index] * ux
    Ty = p.phase_thrust[phase_index] * uy
    Tz = p.phase_thrust[phase_index] * uz

    # Drag force
    drag_factor = -(0.5 * p.drag_linear_density) * casadi.exp(-(r - 1)/p.scale_height) * vr
    Dx = drag_factor * vr_x
    Dy = drag_factor * vr_y
    Dz = drag_factor * vr_z

    # Acceleration
    ax = gx + (Dx + Tx) / mass
    ay = gy + (Dy + Ty) / mass
    az = gz + (Dz + Tz) / mass

    mocp.set_derivative(mass, -p.phase_massFlow[phase_index])
    mocp.set_derivative(rx, vx)
    mocp.set_derivative(ry, vy)
    mocp.set_derivative(rz, vz)
    mocp.set_derivative(vx, ax)
    mocp.set_derivative(vy, ay)
    mocp.set_derivative(vz, az)

    ## Constraints

    # Duration is positive and bounded
    mocp.add_constraint(duration > 1e-6)
    mocp.add_constraint(duration < p.phase_maximumDuration[phase_index])

    # Mass is positive
    mocp.add_path_constraint(mass > 1e-6)

    # u is a unit vector
    mocp.add_path_constraint(ux**2 + uy**2 + uz**2 == 1.0)

    # The initial mass for each phase/stage is given
    initial_mass = mocp.start(mass)
    mocp.add_constraint(initial_mass == p.phase_initialMass[phase_index])

    variables = locals().copy()
    RocketStagePhase = collections.namedtuple('RocketStagePhase',sorted(list(variables.keys())))
    return RocketStagePhase(**variables)

def get_parameters_SI():
    SRB_wetMass = 19290.0
    SRB_propMass = 17010.0
    SRB_dryMass = SRB_wetMass - SRB_propMass
    SRB_burnTime = 75.2
    SRB_thrust = 628500.0
    SRB_massFlow = SRB_propMass / SRB_burnTime

    first_wetMass = 104380.0
    first_propMass = 95550.0
    first_dryMass = first_wetMass - first_propMass
    first_burnTime = 261.0
    first_thrust = 1083100.0
    first_massFlow = first_propMass / first_burnTime

    second_wetMass = 19300.0
    second_propMass = 16820.0
    second_dryMass = second_wetMass - second_propMass
    second_burnTime = 700.0
    second_thrust = 110094.0
    second_massFlow = second_propMass / second_burnTime

    payload_mass = 4164.0

    phase_thrust = [6*SRB_thrust + first_thrust, 3*SRB_thrust + first_thrust, first_thrust, second_thrust]
    phase_massFlow = [6*SRB_massFlow + first_massFlow, 3*SRB_massFlow + first_massFlow, first_massFlow, second_massFlow]
    phase_initialMass = \
        [9 * SRB_wetMass + first_wetMass + second_wetMass + payload_mass,
         3 * SRB_wetMass + first_wetMass + second_wetMass + payload_mass - SRB_burnTime/first_burnTime * first_propMass,
         first_wetMass + second_wetMass + payload_mass - 2*SRB_burnTime/first_burnTime * first_propMass,
         second_wetMass + payload_mass]
    phase_maximumDuration = [SRB_burnTime, SRB_burnTime, first_burnTime-2*SRB_burnTime, second_burnTime]

    scale_height = 7200.0
    drag_linear_density = 0.5 * 4.0 * pi * 1.225 # in (kg/m), == Cd * S * rho0
    body_rotation_speed = 7.29211585e-5

    variables = locals().copy()
    AscentProblemParametersSI = collections.namedtuple('AscentProblemParametersSI',sorted(list(variables.keys())))
    return AscentProblemParametersSI(**variables)


def get_scale(mass):
    # Convert to: mu_e = 1, g0 = 1, R_e = 1, m0 = 1
    length = 6378145.0
    time = (3.986012e14)**(-1.0/2) * (length)**(3.0/2)
    speed = length / time
    acceleration = speed / time
    force = mass * acceleration
    mass_flow = mass / time
    linear_density = mass / length

    variables = locals().copy()
    Scale = collections.namedtuple('Scale',sorted(list(variables.keys())))
    return Scale(**variables)


def get_parameters():
    SI = get_parameters_SI()
    scale = get_scale(SI.phase_initialMass[0])

    SRB_burnTime           =  SI.SRB_burnTime               /  scale.time
    SRB_dryMass            =  SI.SRB_dryMass                /  scale.mass
    SRB_massFlow           =  SI.SRB_massFlow               /  scale.mass_flow
    SRB_propMass           =  SI.SRB_propMass               /  scale.mass
    SRB_thrust             =  SI.SRB_thrust                 /  scale.force
    SRB_wetMass            =  SI.SRB_wetMass                /  scale.mass
    body_rotation_speed    =  SI.body_rotation_speed        /  (1/scale.time)
    drag_linear_density    =  SI.drag_linear_density        /  scale.linear_density
    first_burnTime         =  SI.first_burnTime             /  scale.time
    first_dryMass          =  SI.first_dryMass              /  scale.mass
    first_massFlow         =  SI.first_massFlow             /  scale.mass_flow
    first_propMass         =  SI.first_propMass             /  scale.mass
    first_thrust           =  SI.first_thrust               /  scale.force
    first_wetMass          =  SI.first_wetMass              /  scale.mass
    payload_mass           =  SI.payload_mass               /  scale.mass
    phase_initialMass      =  [phase_initialMass_i          /  scale.mass             for phase_initialMass_i in SI.phase_initialMass]
    phase_massFlow         =  [phase_massFlow_i             /  scale.mass_flow        for phase_massFlow_i in SI.phase_massFlow]
    phase_maximumDuration  =  [phase_maximumDuration_i      /  scale.time             for phase_maximumDuration_i in SI.phase_maximumDuration]
    phase_thrust           =  [phase_thrust_i               /  scale.force            for phase_thrust_i in SI.phase_thrust]
    scale_height           =  SI.scale_height               /  scale.length
    second_burnTime        =  SI.second_burnTime            /  scale.time
    second_dryMass         =  SI.second_dryMass             /  scale.mass
    second_massFlow        =  SI.second_massFlow            /  scale.mass_flow
    second_propMass        =  SI.second_propMass            /  scale.mass
    second_thrust          =  SI.second_thrust              /  scale.force
    second_wetMass         =  SI.second_wetMass             /  scale.mass


    # Target orbit parameters, based on: sma: 24361140 m;  ecc: 0.7308;  inc: 28.5 deg;  LAN: 269.8 deg;  ArgP: 130.5 deg
    target_angular_momentum = [-0.636535803170402, 0.0022219381389848, 1.17236025261702]
    target_eccentricity = [0.490016528022977, 0.472909037956365, 0.265159356017268]

    # Initial state
    initial_position_x = math.cos(28.5/180*pi)
    initial_position_y = 0.0
    initial_position_z = math.sin(28.5/180*pi)

    initial_velocity_x = -body_rotation_speed * initial_position_y
    initial_velocity_y = body_rotation_speed * initial_position_x
    initial_velocity_z = 0.0

    variables = locals().copy()
    AscentProblemParameters = collections.namedtuple('AscentProblemParameters',sorted(list(variables.keys())))
    return AscentProblemParameters(**variables)


def generate_figures(t_grid, trajectories, p):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    trajectory_scale = dict()
    trajectory_scale['mass'] = p.scale.mass * 1e-5
    trajectory_scale['rx']   = p.scale.length * 1e-6
    trajectory_scale['ry']   = p.scale.length * 1e-6
    trajectory_scale['rz']   = p.scale.length * 1e-6
    trajectory_scale['vx']   = p.scale.speed * 1e-3
    trajectory_scale['vy']   = p.scale.speed * 1e-3
    trajectory_scale['vz']   = p.scale.speed * 1e-3
    trajectory_scale['ux']   = 1.0
    trajectory_scale['uy']   = 1.0
    trajectory_scale['uz']   = 1.0

    trajectory_ylabel = dict()
    trajectory_ylabel['mass'] = 'Mass (100,000 kg)'
    trajectory_ylabel['rx']   = 'Length (1000 km)'
    trajectory_ylabel['ry']   = 'Length (1000 km)'
    trajectory_ylabel['rz']   = 'Length (1000 km)'
    trajectory_ylabel['vx']   = 'Speed (km/s)'
    trajectory_ylabel['vy']   = 'Speed (km/s)'
    trajectory_ylabel['vz']   = 'Speed (km/s)'
    trajectory_ylabel['ux']   = ''
    trajectory_ylabel['uy']   = ''
    trajectory_ylabel['uz']   = ''

    for trajectory_name in trajectories:
        fig, ax = plt.subplots()
        ax.plot(p.scale.time * np.array(t_grid), trajectory_scale[trajectory_name] * np.array(trajectories[trajectory_name]))
        ax.set(xlabel='Time (s)', ylabel=trajectory_ylabel[trajectory_name], title=trajectory_name)
        ax.grid()
        fig.savefig('Titan3_' + trajectory_name + '.png', dpi=288)


if __name__ == '__main__':
    main()