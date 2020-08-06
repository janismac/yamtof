import casadi
from yamtof.mocp import MultiPhaseOptimalControlProblem as MOCP
from yamtof import mocp
from yamtof import collocation
import collections

# MocpTranscriber
#
# * Transcribes the MOCP into a standard NLP using a local collocation method
# * Packs and unpacks the variable and parameter vector
#
# standard form:
# min   f(x,p)
# s.t.  h(x,p) == 0
#       g(x,p) <= 0
#        lb(p) <= x <= ub(p)
#
class MocpTranscriber:
    def __init__(self, _mocp):
        assert isinstance(_mocp, MOCP)
        self.mocp = _mocp

        self.phases = dict()
        for phase_name in self.mocp.phases:
            self.phases[phase_name] = TranscribedPhase(self.mocp.phases[phase_name])

        # Create overall objective
        total_objective_sym = casadi.SX(0.0)
        total_objective_sym += sum(self.mocp.objectives.values())

        for phase_name in self.phases:
            total_objective_sym += sum(self.phases[phase_name].mean_objective_integrals.values())

        # Collect all constraints
        equality_constraints = []    # every entry == 0
        inequality_constraints = []  # every entry <= 0

        # ODE constraints
        for phase_name in self.phases:
            for trajectory_name in self.phases[phase_name].trajectories:
                ode_node_defects = self.phases[phase_name].trajectories[trajectory_name].ode_node_defects
                if not ode_node_defects is None:
                    equality_constraints.append(ode_node_defects)

        # Path constraints
        for phase_name in self.phases:
            equality_constraints.extend(c.g_nodes for c in self.phases[phase_name].constraints.values() if c.ocp_constraint.is_equation)
            inequality_constraints.extend(c.g_nodes for c in self.phases[phase_name].constraints.values() if not c.ocp_constraint.is_equation)

        # Other constraints
        equality_constraints.extend(c.g for c in self.mocp.constraints.values() if c.is_equation)
        inequality_constraints.extend(c.g for c in self.mocp.constraints.values() if not c.is_equation)

        equality_constraints = casadi.vertcat(*equality_constraints)
        inequality_constraints = casadi.vertcat(*inequality_constraints)

        # Sort box constraints
        equality_constraints_box = []
        equality_constraints_other = []
        inequality_constraints_box = []
        inequality_constraints_other = []

        for i in range(inequality_constraints.numel()):
            b = self.parse_box_constraint(inequality_constraints[i])
            if b is None:
                inequality_constraints_other.append(inequality_constraints[i])
            else:
                inequality_constraints_box.append(b)

        for i in range(equality_constraints.numel()):
            b = self.parse_box_constraint(equality_constraints[i])
            if b is None:
                equality_constraints_other.append(equality_constraints[i])
            else:
                equality_constraints_box.append(b)

        equality_constraints_other = casadi.vertcat(*equality_constraints_other)
        inequality_constraints_other = casadi.vertcat(*inequality_constraints_other)

        # Combine box constraints into vectors matching the variables (x) vector
        x_symbol, x_value = self.pack_variables()
        x_names = [e.name() for e in casadi.vertsplit(x_symbol)]
        assert len(list(set(x_names))) == len(x_names), 'Duplicate variable name!'
        x_name_index_map = {e[1]:e[0] for e in enumerate(x_names)}
        lower_bound = [casadi.SX(-casadi.inf)] * x_symbol.numel()
        upper_bound = [casadi.SX(casadi.inf)] * x_symbol.numel()

        for i in range(len(inequality_constraints_box)):
            variable_index = x_name_index_map[inequality_constraints_box[i].variable.name()]
            if inequality_constraints_box[i].is_upper_bound:
                upper_bound[variable_index] = casadi.fmin(upper_bound[variable_index], inequality_constraints_box[i].bound)
            else:
                lower_bound[variable_index] = casadi.fmax(lower_bound[variable_index], inequality_constraints_box[i].bound)

        for i in range(len(equality_constraints_box)):
            variable_index = x_name_index_map[equality_constraints_box[i].variable.name()]
            upper_bound[variable_index] = casadi.fmin(upper_bound[variable_index], equality_constraints_box[i].bound)
            lower_bound[variable_index] = casadi.fmax(lower_bound[variable_index], equality_constraints_box[i].bound)

        lower_bound = casadi.vertcat(*lower_bound)
        upper_bound = casadi.vertcat(*upper_bound)

        # Create standard form f,h,g,lb,ub functions
        p_symbol, p_value = self.pack_parameters()
        self.nlp_x = x_symbol
        self.nlp_p = p_symbol
        self.nlp_f = total_objective_sym
        self.nlp_h = equality_constraints_other
        self.nlp_g = inequality_constraints_other
        self.nlp_lb = lower_bound
        self.nlp_ub = upper_bound

        self.nlp_fn_fhg = casadi.Function(
            'nlp_fn_fhg',
            [self.nlp_x, self.nlp_p],
            [self.nlp_f, self.nlp_h, self.nlp_g],
            ['x','p'],
            ['f','h','g'])

        self.nlp_fn_lb_ub = casadi.Function(
            'nlp_fn_lb_ub',
            [self.nlp_p],
            [self.nlp_lb, self.nlp_ub],
            ['p'],
            ['lb','ub'])

        assert not self.nlp_fn_fhg.has_free()
        assert not self.nlp_fn_lb_ub.has_free()

    def parse_box_constraint(self, g):
        assert(g.numel() == 1)
        op = g.op()
        if (not g.is_symbolic()) and (not op in (casadi.OP_SUB, casadi.OP_NEG)):
            raise NotImplementedError # All constraints are currently implemented as: g == lhs - rhs

        # Parse: g == lhs - rhs
        if op == casadi.OP_SUB:
            lhs = g.dep(0)
            rhs = g.dep(1)
        elif op == casadi.OP_NEG:
            lhs = casadi.SX(0.0)
            rhs = g.dep(0)
        elif g.is_symbolic():
            lhs = g
            rhs = casadi.SX(0.0)

        if not (lhs.is_leaf() and rhs.is_leaf()): return None # Not a (simple) box constraint

        if lhs.is_constant():
            is_lhs_variable = False
        else:
            assert lhs.is_symbolic()
            is_lhs_variable = not lhs.name().startswith('P')

        if rhs.is_constant():
            is_rhs_variable = False
        else:
            assert rhs.is_symbolic()
            is_rhs_variable = not rhs.name().startswith('P')

        if is_lhs_variable and is_rhs_variable: return None # Not a box constraint

        if (not is_lhs_variable) and (not is_rhs_variable):
            raise RuntimeError('A constraint without variables is non-sense')

        # g <= 0
        # lhs - rhs <= 0
        # lhs <= rhs

        # is_rhs_variable => lhs is lower bound
        # is_lhs_variable => rhs is upper bound

        BoxConstraint = collections.namedtuple('BoxConstraint',['variable', 'bound', 'is_upper_bound'])

        if is_lhs_variable:
            return BoxConstraint(variable=lhs, bound=rhs, is_upper_bound=True)
        else:
            return BoxConstraint(variable=rhs, bound=lhs, is_upper_bound=False)

    def pack_variables(self):
        packed_symbols = []
        packed_values = []
        for phase_name in self.mocp.phases:
            for trajectory_name in self.mocp.phases[phase_name].trajectories:
                node_symbols = self.phases[phase_name].trajectories[trajectory_name].node_symbols
                node_values = self.mocp.phases[phase_name].trajectories[trajectory_name].values
                assert len(node_symbols) == len(node_values)
                packed_symbols.extend(node_symbols)
                packed_values.extend(node_values)
            packed_symbols.append(self.mocp.phases[phase_name].duration_symbol)
            packed_values.append(self.mocp.phases[phase_name].duration_value)

        for variable_name in self.mocp.variables:
            packed_symbols.append(self.mocp.variables[variable_name].symbol)
            packed_values.append(self.mocp.variables[variable_name].value)

        packed_symbols = casadi.vertcat(casadi.SX(), *packed_symbols)
        packed_values = casadi.vertcat(casadi.DM(), *packed_values)
        assert isinstance(packed_symbols, casadi.SX)
        assert isinstance(packed_values, casadi.DM)
        return (packed_symbols, packed_values)

    def unpack_variables(self, x_value):
        assert isinstance(x_value, casadi.DM)
        j = 0
        for phase_name in self.mocp.phases:
            for trajectory_name in self.mocp.phases[phase_name].trajectories:
                for i in range(len(self.mocp.phases[phase_name].trajectories[trajectory_name].values)):
                    self.mocp.phases[phase_name].trajectories[trajectory_name].values[i] = float(x_value[j])
                    j += 1

            self.mocp.phases[phase_name].duration_value = float(x_value[j])
            j += 1

        for variable_name in self.mocp.variables:
            self.mocp.variables[variable_name].value = float(x_value[j])
            j += 1

        assert x_value.numel() == j

    def pack_parameters(self):
        packed_symbols = []
        packed_values = []

        for parameter_name in self.mocp.parameters:
            packed_symbols.append(self.mocp.parameters[parameter_name].symbol)
            packed_values.append(self.mocp.parameters[parameter_name].value)

        packed_symbols = casadi.vertcat(casadi.SX(), *packed_symbols)
        packed_values = casadi.vertcat(casadi.DM(), *packed_values)
        assert isinstance(packed_symbols, casadi.SX)
        assert isinstance(packed_values, casadi.DM)
        return (packed_symbols, packed_values)

class TranscribedPhase:
    def __init__(self, ocp_phase):
        assert isinstance(ocp_phase, mocp.MOCP_Phase)
        self.ocp_phase = ocp_phase
        self.mean_objective_integrals = dict()

        # Make trajectory node symbols
        self.trajectories = dict()
        for trajectory_name in ocp_phase.trajectories:
            self.trajectories[trajectory_name] = TranscribedTrajectory(ocp_phase.trajectories[trajectory_name])

        # Substitution pairs from iterior to node
        n_phase_nodes = self.ocp_phase.n_intervals * (collocation.n_nodes - 1) + 1
        self.node_substitutions = [None] * n_phase_nodes
        for i in range(n_phase_nodes):
            self.node_substitutions[i] = (
                [ocp_phase.trajectories[trajectory_name].trajectory_interior for trajectory_name in ocp_phase.trajectories],
                [self.trajectories[trajectory_name].node_symbols[i] for trajectory_name in ocp_phase.trajectories]
            )

        # Create ODE constraints
        for trajectory_name in ocp_phase.trajectories:
            self.trajectories[trajectory_name].create_ode_constraints(self)

        # Path constraint instances on nodes
        self.constraints = dict()
        for constraint_name in ocp_phase.constraints:
            self.constraints[constraint_name] = TranscribedPathConstraint(ocp_phase.constraints[constraint_name], self)

        # Create mean/integral objectives
        quadrature_weights = casadi.DM(collocation.integration_weights)/self.ocp_phase.n_intervals
        for objective_name in self.ocp_phase.mean_objectives:
            dF_dtau = self.ocp_phase.mean_objectives[objective_name]
            dF_dtau_nodes = self.substitute_nodes(dF_dtau)

            interval_means = []
            for i in range(self.ocp_phase.n_intervals):
                interval_slice = [i * (collocation.n_nodes-1) + k for k in range(collocation.n_nodes)]
                interval_means.append(quadrature_weights.T @ casadi.vertcat(*[dF_dtau_nodes[i] for i in interval_slice]))

            self.mean_objective_integrals[objective_name] = sum(interval_means)

    def substitute_nodes(self, f):
        return [casadi.substitute([f], sub[0], sub[1])[0] for sub in self.node_substitutions]

class TranscribedTrajectory:
    def __init__(self, ocp_trajectory):
        assert isinstance(ocp_trajectory, mocp.MOCP_Trajectory)
        self.ocp_trajectory = ocp_trajectory
        n_phase_nodes = len(ocp_trajectory.values)
        n_intervals = ocp_trajectory.parent_phase.n_intervals
        assert n_phase_nodes == (n_intervals * (collocation.n_nodes - 1) + 1)
        sx_trajectory = ocp_trajectory.trajectory_interior
        sx_name_parts = mocp.parse_variable_name(sx_trajectory.name())
        self.node_symbols = [None] * n_phase_nodes

        for i in range(1,n_phase_nodes-1):
            self.node_symbols[i] = casadi.SX.sym('N/' + sx_name_parts.phase + '/' + sx_name_parts.name + '/' + str(i))

        self.node_symbols[0]  = ocp_trajectory.start
        self.node_symbols[-1] = ocp_trajectory.end

        self.ode_node_defects = None

    def create_ode_constraints(self, transcribed_phase):
        if self.ocp_trajectory.derivative is None: return None
        assert isinstance(self.ocp_trajectory.derivative, casadi.SX)
        assert self.ocp_trajectory.derivative.numel() == 1

        n_intervals = transcribed_phase.ocp_phase.n_intervals
        node_derivatives = transcribed_phase.substitute_nodes(self.ocp_trajectory.derivative)
        interval_duration = transcribed_phase.ocp_phase.duration_symbol / n_intervals
        defects = []
        for i in range(n_intervals):
            interval_slice = [i * (collocation.n_nodes-1) + k for k in range(collocation.n_nodes)]

            dxdt_interval = [node_derivatives[j] for j in interval_slice]
            x_interval = [self.node_symbols[j] for j in interval_slice]
            delta_x_interval = [(xi - x_interval[0])  for xi in x_interval[1:]]

            dxdt_integrals = interval_duration * (casadi.DM(collocation.integration_matrix) @ casadi.vertcat(*dxdt_interval))
            defects_interval = dxdt_integrals - casadi.vertcat(*delta_x_interval)
            defects.append(defects_interval)

        self.ode_node_defects = casadi.vertcat(*defects)

class TranscribedPathConstraint:
    def __init__(self, ocp_constraint, transcribed_phase):
        assert isinstance(ocp_constraint, mocp.MOCP_Constraint)
        assert isinstance(ocp_constraint.phase_name, str)
        assert isinstance(ocp_constraint.parent_phase, mocp.MOCP_Phase)
        assert isinstance(transcribed_phase, TranscribedPhase)
        self.ocp_constraint = ocp_constraint
        self.g_nodes = casadi.vertcat(*transcribed_phase.substitute_nodes(ocp_constraint.g))