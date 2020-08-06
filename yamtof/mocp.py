from yamtof import collocation
import re
import casadi
import inspect
import collections

# symbolic variable naming scheme:
# flag/phase/name
# flag/phase/name[/index]

# flags:
# D - phase durations
# T - trajectory interior
# S - trajectory start
# E - trajectory end
# V - phaseless variable
# P - phaseless parameter
VariableInfo = collections.namedtuple('VariableInfo',['flag','phase','name','index'])


class MOCP_Trajectory:
    def __init__(self, phase_name, trajectory_name, n_intervals, initialization_value, parent_phase):
        assert isinstance(phase_name, str)
        assert isinstance(parent_phase, MOCP_Phase)
        assert is_valid_name(phase_name)
        assert isinstance(trajectory_name, str)
        assert is_valid_name(trajectory_name)
        assert isinstance(n_intervals, int)
        assert n_intervals >= 1
        if not isinstance(initialization_value, float): initialization_value = 0.0

        self.trajectory_interior = casadi.SX.sym('T/' + phase_name + '/' + trajectory_name)
        self.start               = casadi.SX.sym('S/' + phase_name + '/' + trajectory_name)
        self.end                 = casadi.SX.sym('E/' + phase_name + '/' + trajectory_name)
        self.derivative          = None
        self.values              = [initialization_value] * (n_intervals * (collocation.n_nodes - 1) + 1)
        self.parent_phase        = parent_phase

    # TODO set/assign timeseries

class MOCP_Scalar:
    def __init__(self, sym_name, value):
        assert isinstance(sym_name, str)
        assert isinstance(value, float)
        self.symbol = casadi.SX.sym(sym_name)
        self.value = value

class MOCP_Phase:
    def __init__(self, phase_name, n_intervals, duration_value, parent_mocp):
        assert isinstance(phase_name, str)
        assert is_valid_name(phase_name)
        assert isinstance(n_intervals, int)
        assert n_intervals >= 1
        assert isinstance(parent_mocp, MultiPhaseOptimalControlProblem)

        self.phase_name          = phase_name
        self.trajectories        = dict()
        self.parent_mocp         = parent_mocp
        self.n_intervals         = n_intervals
        self.duration_symbol     = casadi.SX.sym('D/' + phase_name + '/')
        self.duration_value      = duration_value
        self.constraints         = dict()
        self.mean_objectives     = dict()
        self.outputs             = dict()

    def add_trajectory(self, trajectory_name, initialization_value):
        assert isinstance(trajectory_name, str)
        assert is_valid_name(trajectory_name)
        assert trajectory_name not in self.trajectories
        self.trajectories[trajectory_name] = MOCP_Trajectory(self.phase_name, trajectory_name, self.n_intervals, initialization_value, self)
        return self.trajectories[trajectory_name]

    def change_time_resolution(self, n_intervals_new):
        M = make_phase_interpolation_matrix(self.n_intervals, make_phase_nodes(n_intervals_new))
        for trajectory_name in self.trajectories:
            self.trajectories[trajectory_name].values = [float(f) for f in casadi.vertsplit(M @ self.trajectories[trajectory_name].values)]
        self.n_intervals = n_intervals_new

    def interpolate(self, tau_evaluation_points):
        M = make_phase_interpolation_matrix(self.n_intervals, tau_evaluation_points)

        result = dict()
        result['duration'] = self.duration_value
        result['trajectories'] = dict()
        result['outputs'] = dict()

        for trajectory_name in self.trajectories:
            result['trajectories'][trajectory_name] = [float(f) for f in casadi.vertsplit(M @ self.trajectories[trajectory_name].values)]

        pairs = self.parent_mocp.get_symbol_value_pairs()
        pairs = list(zip(*pairs))
        non_path_symbols = casadi.vertcat(*pairs[0])
        non_path_values = casadi.DM(pairs[1])

        trajectroy_interiors = casadi.vertcat(*[e.trajectory_interior for e in self.trajectories.values()])

        outputs_fn = casadi.Function(
            'outputs_fn',
            [non_path_symbols, trajectroy_interiors],
            list(self.outputs.values()),
            ['non_path_symbols','trajectroy_interiors'],
            list(self.outputs.keys())).map(tau_evaluation_points.size1())

        output_values = outputs_fn.call({'non_path_symbols':non_path_values, 'trajectroy_interiors':casadi.DM([e for e in result['trajectories'].values()])})
        for output_name in output_values:
            result['outputs'][output_name] = [float(f) for f in casadi.vertsplit(output_values[output_name].T)]

        return result

    def add_output(self, name, fn):
        self.outputs[name] = fn

# MOCP_Constraint represents either a simple constraint or a path constraint.
# MOCP_Constraint represents either an equality or an inequality constraint.
#     is_equation and     is_path_constraint =>  g(t) == 0 for all t
#     is_equation and not is_path_constraint =>     g == 0
# not is_equation and     is_path_constraint =>  g(t) <= 0 for all t
# not is_equation and not is_path_constraint =>     g <= 0
class MOCP_Constraint:
    def __init__(self, name, is_equation, is_path_constraint, g):
        assert isinstance(g, casadi.SX)
        assert g.numel() == 1

        self.name = name
        self.is_equation = is_equation
        self.is_path_constraint = is_path_constraint
        self.g = g
        self.phase_name = None
        self.parent_phase = None

        constraint_variables = get_all_SX_leaves(g)
        _has_trajectory_variable = has_trajectory_variable(constraint_variables)
        phase_names = get_unique_phase_names(constraint_variables)

        if self.is_path_constraint:
            assert _has_trajectory_variable, 'Error: A path constraint must use a trajectory variable.'
            assert len(phase_names) == 1, 'Error: A path constraint must apply to one and only one phase.'
            self.phase_name = phase_names[0]
        else:
            assert not _has_trajectory_variable, 'Error: A simple constraint must not use trajectory variables. Use a path constraint.'

class MultiPhaseOptimalControlProblem:
    def __init__(self):
        self.phases = dict()
        self.variables = dict()
        self.parameters = dict()
        self.constraints = dict()
        self.objectives = dict()

    def create_phase(self, phase_name, **kwargs):
        assert not phase_name in self.phases
        init = kwargs.get('init', 1.0)
        n_intervals = kwargs.get('n_intervals', 2)
        self.phases[phase_name] = MOCP_Phase(phase_name, n_intervals, init, self)
        return self.phases[phase_name].duration_symbol

    def add_trajectory(self, phase_name, trajectory_name, **kwargs):
        init = kwargs.get('init', 0.0)
        assert isinstance(trajectory_name, str)
        assert is_valid_name(trajectory_name)
        if not phase_name in self.phases: self.create_phase(phase_name)
        assert not trajectory_name in self.phases[phase_name].trajectories
        self.phases[phase_name].add_trajectory(trajectory_name, init)
        return self.phases[phase_name].trajectories[trajectory_name].trajectory_interior

    def get_phase_duration(self, phase_name):
        if not phase_name in self.phases: self.create_phase(phase_name)
        return self.phases[phase_name].duration_symbol

    def add_variable(self, name, **kwargs):
        init = kwargs.get('init', 0.0)
        assert isinstance(name, str)
        assert is_valid_name(name)
        assert name not in self.variables
        self.variables[name] = MOCP_Scalar('V//' + name, init)
        return self.variables[name].symbol

    def add_parameter(self, name, **kwargs):
        init = kwargs.get('init', 0.0)
        assert isinstance(name, str)
        assert is_valid_name(name)
        assert name not in self.parameters
        self.parameters[name] = MOCP_Scalar('P//' + name, init)
        return self.parameters[name].symbol

    def get_parameter(self, name):
        assert name in self.parameters
        return self.parameters[name].symbol

    def add_path_output(self, name, fn):
        assert isinstance(name, str)
        assert isinstance(fn, casadi.SX)
        assert fn.numel() == 1
        phase_names = list(set(parse_variable_name(n.name()).phase for n in get_all_SX_leaves(fn)))
        phase_names = [n for n in phase_names if not n is None]
        assert len(phase_names) == 1, 'Mixed phases!'
        self.phases[phase_names[0]].add_output(name, fn)

    def inital(self, sx_trajectory): return self.start(sx_trajectory)
    def start(self, sx_trajectory): return self._get_boundary(sx_trajectory, 'start')
    def final(self, sx_trajectory): return self.end(sx_trajectory)
    def end(self, sx_trajectory): return self._get_boundary(sx_trajectory, 'end')

    def get_value(self, sx): return self.access_value(sx)
    def set_value(self, sx, value):
        assert isinstance(value, float)
        self.access_value(sx, value)

    def access_value(self, sx, new_value=None):
        assert isinstance(sx, casadi.SX)
        assert sx.numel() == 1
        assert sx.is_symbolic()
        name_parts = parse_variable_name(sx.name())

        if name_parts.flag == 'T':
            raise RuntimeError('Paths are evaluated using MOCP_Phase.interpolate()')

        assert name_parts.flag in ('D', 'S', 'E', 'V', 'P')

        if name_parts.flag == 'P':
            if isinstance(new_value, float): self.parameters[name_parts.name].value = new_value
            return self.parameters[name_parts.name].value
        elif name_parts.flag == 'V':
            if isinstance(new_value, float): self.variables[name_parts.name].value = new_value
            return self.variables[name_parts.name].value
        elif name_parts.flag == 'D':
            if isinstance(new_value, float): self.phases[name_parts.phase].duration_value = new_value
            return self.phases[name_parts.phase].duration_value
        elif name_parts.flag == 'S':
            if isinstance(new_value, float): self.phases[name_parts.phase].trajectories[name_parts.name].values[0] = new_value
            return self.phases[name_parts.phase].trajectories[name_parts.name].values[0]
        elif name_parts.flag == 'E':
            if isinstance(new_value, float): self.phases[name_parts.phase].trajectories[name_parts.name].values[-1] = new_value
            return self.phases[name_parts.phase].trajectories[name_parts.name].values[-1]

        raise RuntimeError('This line should never be reached')

    def get_symbol_value_pairs(self):
        return \
        [(e.symbol, e.value) for e in self.variables.values()] + \
        [(e.symbol, e.value) for e in self.parameters.values()] + \
        [(e.duration_symbol,e.duration_value) for e in self.phases.values()] + \
        [(f.start, f.values[0]) for e in self.phases.values() for f in e.trajectories.values()] + \
        [(f.end, f.values[-1]) for e in self.phases.values() for f in e.trajectories.values()]

    def _get_boundary(self, sx_trajectory, boundary_name):
        assert isinstance(sx_trajectory, casadi.SX)
        assert sx_trajectory.is_leaf()
        assert sx_trajectory.numel() == 1
        var_name = parse_variable_name(sx_trajectory.name())
        assert var_name.flag == 'T'
        if boundary_name == 'start':
            result = self.phases[var_name.phase].trajectories[var_name.name].start
        else:
            result = self.phases[var_name.phase].trajectories[var_name.name].end
        assert isinstance(result, casadi.SX)
        return result

    def set_derivative(self, x, dxdt_fn):
        if isinstance(dxdt_fn, float): dxdt_fn = casadi.SX(dxdt_fn)
        assert isinstance(x, casadi.SX)
        assert isinstance(dxdt_fn, casadi.SX)
        assert x.is_leaf()
        assert x.numel() == 1
        assert dxdt_fn.numel() == 1
        var_name = parse_variable_name(x.name())
        assert var_name.flag == 'T'
        assert self.phases[var_name.phase].trajectories[var_name.name].derivative is None
        self.phases[var_name.phase].trajectories[var_name.name].derivative = dxdt_fn

    def _add_constraint_impl(self, constraint_expr, name, is_path_constraint):
        assert isinstance(constraint_expr, casadi.SX)
        assert constraint_expr.numel() == 1
        op = constraint_expr.op()
        assert op in (casadi.OP_LE, casadi.OP_LT, casadi.OP_EQ)

        # rewrite constraint: if (is_equation): (g == 0) else: (g < 0)
        is_equation = (op == casadi.OP_EQ)
        lhs = constraint_expr.dep(0)
        rhs = constraint_expr.dep(1)
        g = (lhs - rhs)

        constraint = MOCP_Constraint(name, is_equation, is_path_constraint, g)

        if is_path_constraint:
            assert not name in self.phases[constraint.phase_name].constraints, 'Duplicate name'
            constraint.parent_phase = self.phases[constraint.phase_name]
            self.phases[constraint.phase_name].constraints[name] = constraint
        else:
            assert not name in self.constraints, 'Duplicate name'
            self.constraints[name] = constraint

    def add_constraint(self, constraint_expr, **kwargs):
        if not 'name' in kwargs:
            caller_info = inspect.getouterframes(inspect.currentframe())[1]
            kwargs['name'] = make_name_from_caller_info(caller_info)
        self._add_constraint_impl(constraint_expr, kwargs['name'], False)

    def add_path_constraint(self, constraint_expr, **kwargs):
        if not 'name' in kwargs:
            caller_info = inspect.getouterframes(inspect.currentframe())[1]
            kwargs['name'] = make_name_from_caller_info(caller_info)
        self._add_constraint_impl(constraint_expr, kwargs['name'], True)

    def add_objective(self, f, **kwargs):
        if not 'name' in kwargs:
            caller_info = inspect.getouterframes(inspect.currentframe())[1]
            kwargs['name'] = make_name_from_caller_info(caller_info)
        variables = get_all_SX_leaves(f)
        assert not has_trajectory_variable(variables), 'Error: Simple objectives may not contain a trajectory variable. Use an integral objective.'
        assert not kwargs['name'] in self.objectives, 'Duplicate name'
        self.objectives[kwargs['name']] = f

    def add_integral_objective(self, f, **kwargs):
        if not 'name' in kwargs:
            caller_info = inspect.getouterframes(inspect.currentframe())[1]
            kwargs['name'] = make_name_from_caller_info(caller_info)
        variables = get_all_SX_leaves(f)
        assert has_trajectory_variable(variables), 'Error: Integral objectives must use a trajectory variable.'
        phase_names = get_unique_phase_names(variables)
        assert len(phase_names) == 1, 'Error: Integral objectives must apply to one and only one phase.'
        assert not kwargs['name'] in self.phases[phase_names[0]].mean_objectives, 'Duplicate name'
        self.phases[phase_names[0]].mean_objectives[kwargs['name']] = (self.phases[phase_names[0]].duration_symbol * f)

    def add_mean_objective(self, f, **kwargs):
        if not 'name' in kwargs:
            caller_info = inspect.getouterframes(inspect.currentframe())[1]
            kwargs['name'] = make_name_from_caller_info(caller_info)
        variables = get_all_SX_leaves(f)
        assert has_trajectory_variable(variables), 'Error: Mean objectives must use a trajectory variable.'
        phase_names = get_unique_phase_names(variables)
        assert len(phase_names) == 1, 'Error: Mean objectives must apply to one and only one phase.'
        assert not kwargs['name'] in self.phases[phase_names[0]].mean_objectives, 'Duplicate name'
        self.phases[phase_names[0]].mean_objectives[kwargs['name']] = f

    # Convenience function to transcribe and solve the problem with a default approach and configuration.
    def solve(self, **kwargs):
        from yamtof import mocp_transcriber
        from yamtof import mocp_nlp_solver
        transcriber = mocp_transcriber.MocpTranscriber(self)
        x_symbol, x_value = transcriber.pack_variables()
        p_symbol, p_value = transcriber.pack_parameters()
        solver = mocp_nlp_solver.MocpNlpSolver(transcriber)
        result = solver.solver_loop(x_value, p_value, **kwargs)
        x_value = result[-1]['result']['x']
        transcriber.unpack_variables(x_value)
        slack_values = [(e[0].name(), float(e[1])) for e in zip(casadi.vertsplit(x_symbol), casadi.vertsplit(x_value)) if '/slack' in e[0].name()]
        return (result, slack_values)


### Various helper functions ###

def has_trajectory_variable(sx_list):
    return any([parse_variable_name(v.name()).flag == 'T' for v in sx_list])

def get_unique_phase_names(sx_list):
    phase_names = [parse_variable_name(v.name()).phase for v in sx_list]
    phase_names = [n for n in phase_names if not n is None]
    phase_names = list(set(phase_names))
    return phase_names

def parse_variable_name(name):
    parts = name.split('/')
    assert len(parts) >= 3
    flag = parts[0]
    assert len(flag) == 1
    phase = None
    var_name = None
    index = None
    if len(parts[1]) > 0: phase = parts[1]
    if len(parts[2]) > 0: var_name = parts[2]
    if len(parts) > 3 and len(parts[3]) > 0: index = parts[3]
    return VariableInfo(flag, phase, var_name, index)

def is_valid_name(s):
    return not re.fullmatch('\\w+', s) is None

__generated_names_count = dict()

def make_name_from_caller_info(caller_info):
    filename = caller_info.filename.split('/')[-1]
    if filename.endswith('.py'): filename = filename[:-3]
    name = ''.join(caller_info.code_context)
    name = re.sub('\\s+', ' ', name).strip()
    name = name.replace('<','LEQ')
    name = name.replace('>','GEQ')
    name = name.replace('==','EQ')
    name = re.sub('\\W+', ' ', name).strip()
    name = re.sub('\\s+', '_', name)
    name = re.sub('.*?add.*?constraint', '', name)
    name = re.sub('.*?add.*?objective', '', name)
    name = filename + '_L' + str(caller_info.lineno) + '_' + name
    name = name.replace('__','_')

    # Make sure generated names are not reused. Postfix a counter
    global __generated_names_count

    if name in __generated_names_count:
        __generated_names_count[name] += 1
        name += '_' + str(__generated_names_count[name])
    else:
        __generated_names_count[name] = 1

    return name

def SX_equal(a,b):
    try:
        result = bool(a == b)
        return result
    except:
        return False

def get_all_SX_leaves(e):
    all_leaves = get_all_SX_leaves_impl(e)
    unique_leaves = []
    for L1 in all_leaves:
        if not any(SX_equal(L1, L2) for L2 in unique_leaves):
            unique_leaves.append(L1)
    return unique_leaves

def get_all_SX_leaves_impl(e):
    if e.is_symbolic(): return [e]
    result = []
    for i in range(e.n_dep()):
        result += get_all_SX_leaves_impl(e.dep(i))
    return result

def make_phase_nodes(n_intervals):
    assert collocation.nodes[0] == 0.0
    assert collocation.nodes[-1] == 1.0
    nodes = [(n+i)/n_intervals for i in range(n_intervals) for n in collocation.nodes[:-1]]
    nodes.append(1.0)
    return nodes

def make_phase_interpolation_matrix(n_intervals, tau_evaluation_points):
    # TODO cache input-output pairs, avoid re-calculating

    if isinstance(tau_evaluation_points, list):
        tau_evaluation_points = casadi.DM(tau_evaluation_points)
    assert isinstance(tau_evaluation_points, casadi.DM)
    assert tau_evaluation_points.numel() == tau_evaluation_points.size1()
    assert float(casadi.mmin(casadi.diff(tau_evaluation_points) > 0)) == 1.0 # Must be ascending
    assert float(casadi.mmin(tau_evaluation_points)) >= 0.0
    assert float(casadi.mmax(tau_evaluation_points)) <= 1.0

    tau_evaluation_points_scaled = [float(f)*n_intervals for f in casadi.vertsplit(tau_evaluation_points)]
    interval_indices = [int(f) for f in tau_evaluation_points_scaled]
    interval_values = [f-int(f) for f in tau_evaluation_points_scaled]

    if interval_indices[-1] == n_intervals:
        interval_indices[-1] = n_intervals-1
        interval_values[-1] = 1.0

    interval_interpolation_matrix = make_interval_interpolation_matrix(interval_values)
    phase_interpolation_matrix = casadi.DM(tau_evaluation_points.numel(), (n_intervals * (collocation.n_nodes - 1) + 1))

    for i in range(len(interval_values)):
        column_idx = interval_indices[i]*(collocation.n_nodes-1)
        phase_interpolation_matrix[i,(column_idx):(column_idx+collocation.n_nodes)] = casadi.DM(interval_interpolation_matrix[i]).T

    return phase_interpolation_matrix

def make_interval_interpolation_matrix(interval_values):
    # Evaluate Lagrange basis polynomials
    return [[prod(interval_values[i] - collocation.nodes[m] for m in range(collocation.n_nodes) if m != j) * collocation.interpolation_weights[j] for j in range(collocation.n_nodes)] for i in range(len(interval_values))]

def prod(L):
    p = 1.0
    for f in L: p*=f
    return p