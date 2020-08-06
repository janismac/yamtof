assert __name__ == '__main__'
import collections
import casadi
import math
from casadi import pi, sin, cos
from yamtof.mocp import MultiPhaseOptimalControlProblem as MOCP

def main():
    p = get_parameters()
    mocp = MOCP()
    start = lambda a: mocp.start(a)
    end = lambda a: mocp.end(a)
    weight_dynamic_pressure_penalty = mocp.add_parameter('weight_dynamic_pressure_penalty', init=1e-6)
    merlin_exhaust_velocity = mocp.add_parameter('merlin_exhaust_velocity', init=p.merlin_exhaust_velocity)

    phase_ascent = create_rocket_stage_phase(
        mocp, 'ascent', p,
        is_powered = True,
        maxQ = p.maxQ_ascent,
        engine_type = 'backpressure',
        engine_count = 9,
        mdot_max = p.merlin_max_mdot,
        mdot_min = p.merlin_min_mdot,
        exhaust_velocity = merlin_exhaust_velocity,
        exit_area = p.merlin_exit_area,
        exit_pressure = p.merlin_exit_pressure
    )

    phase_separation_S2 = create_rocket_stage_phase(
        mocp, 'separation_S2', p,
        is_powered = False,
        maxQ = p.maxQ_ascent,
    )

    phase_insertion = create_rocket_stage_phase(
        mocp, 'insertion', p,
        is_powered = True,
        maxQ = p.maxQ_ascent,
        engine_type = 'vacuum',
        engine_count = 1,
        mdot_max = p.mvac_max_mdot,
        mdot_min = p.mvac_min_mdot,
        effective_exhaust_velocity = p.mvac_effective_exhaust_velocity
    )

    # Initial state constraint
    mocp.add_constraint(start(phase_ascent.mass) == p.liftoff_mass)
    mocp.add_constraint(start(phase_ascent.rx) == p.launch_site_x)
    mocp.add_constraint(start(phase_ascent.ry) == p.launch_site_y)
    mocp.add_constraint(start(phase_ascent.rz) == p.launch_site_z)
    mocp.add_constraint(start(phase_ascent.vx) == 0.0)
    mocp.add_constraint(start(phase_ascent.vy) == 0.0)
    mocp.add_constraint(start(phase_ascent.vz) == 0.0)

    # Trajectory continuity constraints
    consecutive_phases = (  # (must be topologically sorted)
        (phase_ascent, phase_separation_S2),
        (phase_separation_S2, phase_insertion)
    )

    for pair in consecutive_phases:
        mocp.add_constraint(start(pair[1].rx) == end(pair[0].rx))
        mocp.add_constraint(start(pair[1].ry) == end(pair[0].ry))
        mocp.add_constraint(start(pair[1].rz) == end(pair[0].rz))
        mocp.add_constraint(start(pair[1].vx) == end(pair[0].vx))
        mocp.add_constraint(start(pair[1].vy) == end(pair[0].vy))
        mocp.add_constraint(start(pair[1].vz) == end(pair[0].vz))

    # Masses, staging, timing
    mocp.add_constraint(start(phase_separation_S2.mass) == p.staging_mass)
    mocp.add_constraint(start(phase_insertion.mass) == end(phase_separation_S2.mass))

    separation_coast_time_S2 = 11.0 / p.scale.time # TODO move to parameters
    mocp.add_constraint(phase_separation_S2.duration < separation_coast_time_S2)

    total_mass_at_staging = 210000.0 / p.scale.mass # TODO move to parameters
    mocp.add_path_constraint(phase_ascent.mass > total_mass_at_staging)

    # Target orbit - soft constraints
    # 3 parameter orbit: SMA,ECC,INC
    #
    # h_target_magnitude - h_final_magnitude == 0
    # h_target_z - h_final_z == 0
    # c_target - c_final == 0

    r_final_ECEF = casadi.vertcat(end(phase_insertion.rx), end(phase_insertion.ry), end(phase_insertion.rz))
    v_final_ECEF = casadi.vertcat(end(phase_insertion.vx), end(phase_insertion.vy), end(phase_insertion.vz))

    # Convert to ECI (z rotation is omitted, because LAN is free)
    r_f = r_final_ECEF
    v_f = (v_final_ECEF + casadi.cross(casadi.SX([0,0,p.body_rotation_speed]), r_final_ECEF))

    h_final = casadi.cross(r_f, v_f)
    h_final_mag = casadi.norm_2(h_final)
    h_final_z = h_final[2]
    c_final = casadi.sumsqr(v_f)/2 - 1.0 / casadi.norm_2(r_f)

    defect_h_magnitude = h_final_mag - p.target_orbit_h_magnitude
    defect_h_z = h_final_z - p.target_orbit_h_z
    defect_c = c_final - p.target_orbit_c

    slack_h_magnitude = mocp.add_variable('slack_h_magnitude', init=3.0)
    slack_h_z = mocp.add_variable('slack_h_z', init=3.0)
    slack_c = mocp.add_variable('slack_c', init=3.0)

    mocp.add_constraint(slack_h_magnitude > 0)
    mocp.add_constraint(slack_h_z > 0)
    mocp.add_constraint(slack_c > 0)

    mocp.add_constraint(defect_h_magnitude <  slack_h_magnitude)
    mocp.add_constraint(defect_h_magnitude > -slack_h_magnitude)
    mocp.add_constraint(defect_h_z <  slack_h_z)
    mocp.add_constraint(defect_h_z > -slack_h_z)
    mocp.add_constraint(defect_c <  slack_c)
    mocp.add_constraint(defect_c > -slack_c)

    mocp.add_objective(100.0 * slack_h_magnitude)
    mocp.add_objective(100.0 * slack_h_z)
    mocp.add_objective(100.0 * slack_c)

    # Maximize final mass
    final_mass = end(phase_insertion.mass)
    mocp.add_objective(-1.0 * final_mass)

    ### Problem done, solve it
    mocp.solve(penaltyWeightReductionSteps=20, beta=0.7)
    mocp.add_objective(-10.0 * phase_separation_S2.duration) # Force coast duration up to nominal value
    mocp.set_value(weight_dynamic_pressure_penalty, 1.0) # Force Q violation down
    mocp.solve()

    mocp.phases['ascent'].change_time_resolution(25)
    mocp.phases['separation_S2'].change_time_resolution(4)
    mocp.phases['insertion'].change_time_resolution(20)
    (solver_output, slack_values) = mocp.solve(penaltyWeightReductionSteps=2)

    # Interpolate resulting tajectory
    tau_grid = casadi.linspace(0.0,1.0,501)
    interpolated_results = dict()
    for phase_name in mocp.phases:
        interpolated_results[phase_name] = mocp.phases[phase_name].interpolate(tau_grid)
        interpolated_results[phase_name]['start_time'] = 0.0

    for pair in consecutive_phases: # Fix phase start times
        interpolated_results[pair[1].phase_name]['start_time'] = interpolated_results[pair[0].phase_name]['start_time'] + interpolated_results[pair[0].phase_name]['duration']

    for phase_name in mocp.phases: # Add time grid
        t_gird = interpolated_results[phase_name]['start_time'] + (interpolated_results[phase_name]['duration'] * tau_grid)
        interpolated_results[phase_name]['t'] = [float(f) for f in casadi.vertsplit(t_gird)]

    import json
    with open('falcon9_ascent_ecef_output.json', 'w') as f:
        pairs = mocp.get_symbol_value_pairs()
        json.dump({
            'phases':interpolated_results,
            'scale':p.scale._asdict(),
            'values':{e[0].name():e[1] for e in pairs}
        }, f)

    # Print basic stats
    SECO_mass = mocp.get_value(end(phase_insertion.mass))*p.scale.mass
    SECO_time = (interpolated_results['insertion']['start_time']+interpolated_results['insertion']['duration'])*p.scale.time
    MECO_time = (interpolated_results['ascent']['start_time']+interpolated_results['ascent']['duration'])*p.scale.time
    MECO_mass = mocp.get_value(end(phase_ascent.mass))*p.scale.mass

    print('--------------------------------------------------------')
    print('                              SECO mass            SECO time            MECO mass            MECO time')
    print('Current Result:   ' + f2s(SECO_mass)  + f2s(SECO_time) + f2s(MECO_mass) + f2s(MECO_time))
    print('Previous Result:  ' + f2s(19397.205248) + f2s(513.369232)       + f2s(210000.000049) + f2s(136.830760))
    print('Expected Result:  ' + f2s(1e300*1e300)      + f2s(518.0)     + f2s(1e300*1e300) + f2s(138.0))

    # Check slacks
    max_slack = max([e[1] for e in slack_values])
    assert max_slack < 1e-6



def f2s(f):
    return ("{:21.6f}".format(f))

def create_rocket_stage_phase(mocp, phase_name, p, **kwargs):
    is_powered = kwargs['is_powered']
    maxQ = kwargs['maxQ']

    if is_powered:
        engine_type = kwargs['engine_type']
        engine_count = kwargs['engine_count']
        mdot_max = kwargs['mdot_max']
        mdot_min = kwargs['mdot_min']

        if engine_type == 'vacuum':
            effective_exhaust_velocity = kwargs['effective_exhaust_velocity']
        else:
            exhaust_velocity = kwargs['exhaust_velocity']
            exit_area = kwargs['exit_area']
            exit_pressure = kwargs['exit_pressure']


    duration = mocp.create_phase(phase_name, init=0.001, n_intervals=2)

    # Total vessel mass
    mass  = mocp.add_trajectory(phase_name, 'mass', init=1.0)

    # ECEF position vector
    rx    = mocp.add_trajectory(phase_name, 'rx', init=p.launch_site_x)
    ry    = mocp.add_trajectory(phase_name, 'ry', init=p.launch_site_y)
    rz    = mocp.add_trajectory(phase_name, 'rz', init=p.launch_site_z)
    r_vec = casadi.vertcat(rx,ry,rz)

    # ECEF velocity vector
    vx    = mocp.add_trajectory(phase_name, 'vx', init=0.0)
    vy    = mocp.add_trajectory(phase_name, 'vy', init=0.0)
    vz    = mocp.add_trajectory(phase_name, 'vz', init=0.0)
    v_vec = casadi.vertcat(vx,vy,vz)

    if is_powered:

        # ECEF steering unit vector
        ux    = mocp.add_trajectory(phase_name, 'ux', init=p.launch_site_x)
        uy    = mocp.add_trajectory(phase_name, 'uy', init=p.launch_site_y)
        uz    = mocp.add_trajectory(phase_name, 'uz', init=p.launch_site_z)

        u_vec = casadi.vertcat(ux,uy,uz)

        # Engine mass flow for a SINGLE engine
        mdot = mocp.add_trajectory(phase_name, 'mdot', init=mdot_max)
        mdot_rate = mocp.add_trajectory(phase_name, 'mdot_rate', init=0.0)

    ## Dynamics

    r_squared = rx**2 + ry**2 + rz**2 + 1e-8
    v_squared = vx**2 + vy**2 + vz**2 + 1e-8
    airspeed = v_squared**0.5
    r = r_squared**0.5
    altitude = r - 1
    r3_inv = r_squared**(-1.5)

    mocp.add_path_output('downrange_distance', casadi.asin(casadi.norm_2(casadi.cross(r_vec/r, casadi.SX([p.launch_site_x,p.launch_site_y,p.launch_site_z])))))
    mocp.add_path_output('altitude', altitude)
    mocp.add_path_output('airspeed', airspeed)
    mocp.add_path_output('vertical_speed', (r_vec.T/r)@v_vec)
    mocp.add_path_output('horizontal_speed_ECEF', casadi.norm_2( v_vec - ((r_vec.T/r)@v_vec)*(r_vec/r) ) )

    # Gravitational acceleration (mu=1 is omitted)
    gx = -r3_inv * rx
    gy = -r3_inv * ry
    gz = -r3_inv * rz

    # Atmosphere
    atmosphere_fraction = casadi.exp(-altitude/p.scale_height)
    air_pressure = p.air_pressure_MSL * atmosphere_fraction
    air_density = p.air_density_MSL * atmosphere_fraction
    dynamic_pressure = 0.5 * air_density * v_squared
    mocp.add_path_output('dynamic_pressure', dynamic_pressure)

    if is_powered:
        lateral_airspeed = casadi.sqrt(casadi.sumsqr(v_vec-((v_vec.T@u_vec)*u_vec))+1e-8)
        lateral_dynamic_pressure = 0.5 * air_density * lateral_airspeed * airspeed # Same as Q * sin(alpha), but without trigonometry
        mocp.add_path_output('lateral_dynamic_pressure', lateral_dynamic_pressure)

        mocp.add_path_output('alpha', casadi.acos((v_vec.T@u_vec)/airspeed)*180.0/pi)
        mocp.add_path_output('pitch', 90.0-casadi.acos((r_vec.T/r)@u_vec)*180.0/pi)

    # Thrust force
    if is_powered:
        if engine_type == 'vacuum':
            engine_thrust = effective_exhaust_velocity * mdot
        else:
            engine_thrust = exhaust_velocity * mdot + exit_area * (exit_pressure - air_pressure)

        total_thrust = engine_thrust * engine_count
        mocp.add_path_output('total_thrust', total_thrust)

        Tx = total_thrust * ux
        Ty = total_thrust * uy
        Tz = total_thrust * uz
    else:
        Tx = 0.0
        Ty = 0.0
        Tz = 0.0

    # Drag force
    drag_factor = (-0.5 * p.drag_area) * air_density * airspeed
    Dx = drag_factor * vx
    Dy = drag_factor * vy
    Dz = drag_factor * vz

    # Coriolis Acceleration
    ax_Coriolis =  2 * vy * p.body_rotation_speed
    ay_Coriolis = -2 * vx * p.body_rotation_speed
    az_Coriolis = 0.0

    # Centrifugal Acceleration
    ax_Centrifugal = rx * p.body_rotation_speed**2
    ay_Centrifugal = ry * p.body_rotation_speed**2
    az_Centrifugal = 0.0

    # Acceleration
    ax = gx + ax_Coriolis + ax_Centrifugal + (Dx + Tx) / mass
    ay = gy + ay_Coriolis + ay_Centrifugal + (Dy + Ty) / mass
    az = gz + az_Coriolis + az_Centrifugal + (Dz + Tz) / mass

    if is_powered:
        mocp.set_derivative(mass, -engine_count * mdot)
        mocp.set_derivative(mdot, mdot_rate)
    else:
        mocp.set_derivative(mass, 0.0)

    mocp.set_derivative(rx, vx)
    mocp.set_derivative(ry, vy)
    mocp.set_derivative(rz, vz)
    mocp.set_derivative(vx, ax)
    mocp.set_derivative(vy, ay)
    mocp.set_derivative(vz, az)

    ## Constraints

    # Duration is positive and bounded
    mocp.add_constraint(duration > 0.001)
    mocp.add_constraint(duration < 10.0)

    # Mass is positive
    mocp.add_path_constraint(mass > 1e-6)

    # maxQ soft constraint
    weight_dynamic_pressure_penalty = mocp.get_parameter('weight_dynamic_pressure_penalty')
    slack_maxQ = mocp.add_trajectory(phase_name, 'slack_maxQ', init=10.0)
    mocp.add_path_constraint(slack_maxQ > 0.0)
    mocp.add_mean_objective(weight_dynamic_pressure_penalty * slack_maxQ)
    mocp.add_path_constraint(dynamic_pressure / maxQ < 1.0 + slack_maxQ)

    if is_powered:
        # u is a unit vector
        mocp.add_path_constraint(ux**2 + uy**2 + uz**2 == 1.0)

        # Engine flow limits
        mocp.add_path_constraint(mdot_min < mdot)
        mocp.add_path_constraint(mdot < mdot_max)

        # Throttle rate reduction
        mocp.add_mean_objective(1e-6 * mdot_rate**2)

        # Lateral dynamic pressure penalty
        mocp.add_mean_objective( (lateral_dynamic_pressure/p.maxQ_sin_alpha)**8 )

    variables = locals().copy()
    RocketStagePhase = collections.namedtuple('RocketStagePhase',sorted(list(variables.keys())))
    return RocketStagePhase(**variables)

def get_parameters_SI():

    # Earth
    scale_height = 7640.0
    air_density_MSL = 1.225
    air_pressure_MSL = 101325.0
    body_rotation_speed = 7.2921159e-5

    # Orbit parameters based on ECI coordinates
    target_orbit_h_magnitude = 51.48506083e9
    target_orbit_h_z = 31.95165489e9
    target_orbit_c = -29.9654122996e6

    # Ground positions based on ECEF coordinates
    launch_site_x = 916107.0
    launch_site_y = -5520158.0
    launch_site_z = 3046040.0

    landing_site_x = 920080.0
    landing_site_y = -5523603.0
    landing_site_z = 3038587.0

    # Vehicle aero
    maxQ_ascent = 25000.0
    maxQ_descent = 65000.0
    maxQ_sin_alpha = 4000.0
    drag_area = 5.3

    # Merlin engine
    merlin_exhaust_velocity  = 3076.0
    merlin_exit_area         = 0.7
    merlin_exit_pressure     = 60000.0
    merlin_max_mdot          = 287.0
    merlin_min_mdot          = 162.0

    # Merlin vacuum engine
    mvac_effective_exhaust_velocity  = 3413.0
    mvac_max_mdot                    = 287.0
    mvac_min_mdot                    = 183.0

    # Masses
    liftoff_mass = 552100.0
    staging_mass = 119000.0 # Mass of wet second stage + payload
    landing_mass = 23000.0  # Mass of empty first stage (+ propellant margin)

    variables = locals().copy()
    ParametersSI = collections.namedtuple('ParametersSI',sorted(list(variables.keys())))
    return ParametersSI(**variables)


def get_scale(mass):
    # Convert to: mu_e = 1, g0 = 1, R_e = 1, m0 = 1
    length = 6371008.8
    time = (3.986004418e14)**(-1.0/2) * (length)**(3.0/2)
    speed = length / time
    area = length**2
    acceleration = speed / time
    force = mass * acceleration
    mass_flow = mass / time
    density = mass / (length**3)
    pressure = force / area

    variables = locals().copy()
    Scale = collections.namedtuple('Scale',sorted(list(variables.keys())))
    return Scale(**variables)


def get_parameters():
    SI = get_parameters_SI()
    scale = get_scale(SI.liftoff_mass)

    air_density_MSL                    =  SI.air_density_MSL                   /  scale.density
    air_pressure_MSL                   =  SI.air_pressure_MSL                  /  scale.pressure
    body_rotation_speed                =  SI.body_rotation_speed               /  (1/scale.time)
    drag_area                          =  SI.drag_area                         /  scale.area
    landing_mass                       =  SI.landing_mass                      /  scale.length
    landing_site_x                     =  SI.landing_site_x                    /  scale.length
    landing_site_y                     =  SI.landing_site_y                    /  scale.length
    landing_site_z                     =  SI.landing_site_z                    /  scale.length
    launch_site_x                      =  SI.launch_site_x                     /  scale.length
    launch_site_y                      =  SI.launch_site_y                     /  scale.length
    launch_site_z                      =  SI.launch_site_z                     /  scale.length
    liftoff_mass                       =  SI.liftoff_mass                      /  scale.mass
    maxQ_ascent                        =  SI.maxQ_ascent                       /  scale.pressure
    maxQ_descent                       =  SI.maxQ_descent                      /  scale.pressure
    maxQ_sin_alpha                     =  SI.maxQ_sin_alpha                    /  scale.pressure
    merlin_exhaust_velocity            =  SI.merlin_exhaust_velocity           /  scale.speed
    merlin_exit_area                   =  SI.merlin_exit_area                  /  scale.area
    merlin_exit_pressure               =  SI.merlin_exit_pressure              /  scale.pressure
    merlin_max_mdot                    =  SI.merlin_max_mdot                   /  scale.mass_flow
    merlin_min_mdot                    =  SI.merlin_min_mdot                   /  scale.mass_flow
    mvac_effective_exhaust_velocity    =  SI.mvac_effective_exhaust_velocity   /  scale.speed
    mvac_max_mdot                      =  SI.mvac_max_mdot                     /  scale.mass_flow
    mvac_min_mdot                      =  SI.mvac_min_mdot                     /  scale.mass_flow
    scale_height                       =  SI.scale_height                      /  scale.length
    staging_mass                       =  SI.staging_mass                      /  scale.mass
    target_orbit_h_magnitude           =  SI.target_orbit_h_magnitude          /  (scale.length**2 / scale.time)
    target_orbit_h_z                   =  SI.target_orbit_h_z                  /  (scale.length**2 / scale.time)
    target_orbit_c                     =  SI.target_orbit_c                    /  (scale.length**2 / scale.time**2)

    variables = locals().copy()
    Parameters = collections.namedtuple('Parameters',sorted(list(variables.keys())))
    return Parameters(**variables)


def generate_figures():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import json

    with open('falcon9_ascent_ecef_output.json', 'r') as f:
        data = json.load(f)

    phases = data['phases']
    scale = data['scale']

    paths = \
        sorted(list(set([('trajectories',k) for e in phases.values() for k in e['trajectories'].keys()])))+\
        sorted(list(set([('outputs',k) for e in phases.values() for k in e['outputs'].keys()])))

    scale_map = {
        'airspeed':                  'speed',
        'altitude':                  'length',
        'downrange_distance':        'length',
        'dynamic_pressure':          'pressure',
        'lateral_dynamic_pressure':  'pressure',
        'mass':                      'mass',
        'mdot':                      'mass_flow',
        'rx':                        'length',
        'ry':                        'length',
        'rz':                        'length',
        'total_thrust':              'force',
        'vx':                        'speed',
        'vy':                        'speed',
        'vz':                        'speed',
        'vertical_speed':            'speed',
        'horizontal_speed_ECEF':     'speed',
    }

    for path in paths:
        fig, ax = plt.subplots()
        for phase_name in phases:
            path_scale = 1.0
            if path[1] in scale_map:
                path_scale = scale[scale_map[path[1]]]
                if scale_map[path[1]] == 'length':
                    path_scale /= 1000.0 # show km
                if scale_map[path[1]] == 'force':
                    path_scale /= 1000.0 # show kN
                if scale_map[path[1]] == 'pressure':
                    path_scale /= 1000.0 # show kPa
                if scale_map[path[1]] == 'mass':
                    path_scale /= 1000.0 # show metric tons

            ax.set(xlabel='Time (s)', ylabel=path[1], title=path[1])

            if path[1] in phases[phase_name][path[0]]:
                ax.plot(scale['time'] * np.array(phases[phase_name]['t']),
                    np.array(phases[phase_name][path[0]][path[1]])*path_scale)

        ax.grid()
        fig.savefig('Falcon9_' + path[1] + '.png', dpi=288)
        plt.close('all')

    # downrange_vs_altitude
    fig, ax = plt.subplots()
    length_to_kilometer = scale['length']/1000
    for phase_name in phases:
        ax.plot(
            np.array(phases[phase_name]['outputs']['downrange_distance']) * length_to_kilometer,
            np.array(phases[phase_name]['outputs']['altitude']) * length_to_kilometer)

    ax.set(xlabel='Downrange distance (km)', ylabel='Altitude (km)')
    ax.grid()
    ax.axis('equal')
    fig.savefig('Falcon9_downrange_vs_altitude.png', dpi=288)
    ax.set_xlim(-10,120)
    ax.set_ylim(-10,120)
    fig.savefig('Falcon9_downrange_vs_altitude_zoomed.png', dpi=288)
    plt.close('all')

main()
generate_figures()