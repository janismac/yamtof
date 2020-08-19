import casadi
from yamtof.mocp_transcriber import MocpTranscriber

class MocpNlpSolver:
    def __init__(self, transcriber):
        assert isinstance(transcriber, MocpTranscriber)
        self.transcriber = transcriber

        # Add penalty for deviations from the initial guess
        x_names = [n.name() for n in casadi.vertsplit(transcriber.nlp_x)]
        self.y_indices = [n[0] for n in enumerate(x_names) if '/slack' not in n[1]]
        y_sym = casadi.vertcat(*[transcriber.nlp_x[i] for i in self.y_indices])
        y_init_sym = casadi.vertcat(*[casadi.SX.sym('__init_' + x_names[i]) for i in self.y_indices])

        initPenaltyWeight_sym = casadi.SX.sym('__initPenaltyWeight')
        initPenalty = initPenaltyWeight_sym * casadi.sum1((y_sym - y_init_sym)**2)

        # Inputs for the CasADi nlpsol interface
        self.casadi_lbg = casadi.vertcat(casadi.DM.zeros(transcriber.nlp_h.numel()), (-casadi.inf) * casadi.DM.ones(transcriber.nlp_g.numel()))
        self.casadi_ubg = casadi.vertcat(casadi.DM.zeros(transcriber.nlp_h.numel()), casadi.DM.zeros(transcriber.nlp_g.numel()))
        self.casadi_g   = casadi.vertcat(transcriber.nlp_h, transcriber.nlp_g)

        self.casadi_nlp = {
            'x': transcriber.nlp_x,
            'p': casadi.vertcat(transcriber.nlp_p, initPenaltyWeight_sym, y_init_sym),
            'f': transcriber.nlp_f + initPenalty,
            'g': self.casadi_g
        }

        # https://www.coin-or.org/Bonmin/option_pages/options_list_ipopt.html
        options = {
            'bound_consistency': False ,
            'ipopt': {
                'max_iter': 2000 ,
                #'tol': 1e-06 ,
                'max_cpu_time': 1800 ,
                'mu_strategy': 'adaptive' ,
                'nlp_scaling_method': 'gradient-based' , # 'none', 'user-scaling', 'gradient-based', 'equilibration-based'
                'mumps_permuting_scaling': 7 ,
                'mumps_scaling': 8 ,
                'bound_frac': 1e-06 ,
                'bound_push': 1e-06 ,
                'slack_bound_push': 1e-06 ,
                'bound_relax_factor': 0.0 ,
                'honor_original_bounds': 'no' ,
                'warm_start_init_point': 'yes' ,
                'warm_start_mult_bound_push': 1e-06 ,
                'warm_start_bound_frac': 1e-06 ,
                'warm_start_bound_push': 1e-06 ,
                'warm_start_slack_bound_push': 1e-06 ,
                'warm_start_slack_bound_frac': 1e-06 ,
                'mu_max': 0.01 ,
                'linear_solver': 'mumps'
            }
        }
        self.casadi_nlpsol = casadi.nlpsol('nlpsolver', 'ipopt', self.casadi_nlp, options)

    def solver_step(self, x_value, p_value, initPenaltyWeight, previousResult = None):
        lbxubx = self.transcriber.nlp_fn_lb_ub.call({'p':p_value})

        if previousResult is None:
            result = self.casadi_nlpsol(\
                x0 = x_value, \
                lbx = lbxubx['lb'], \
                ubx = lbxubx['ub'], \
                lbg = self.casadi_lbg, \
                ubg = self.casadi_ubg, \
                p = casadi.vertcat(p_value, casadi.DM(initPenaltyWeight), x_value[self.y_indices])
            )
        else:
            # Reuse duals if available
            assert previousResult['lam_x'].numel() == x_value.numel()
            assert previousResult['lam_g'].numel() == self.casadi_lbg.numel()

            result = self.casadi_nlpsol(\
                x0 = x_value, \
                lbx = lbxubx['lb'], \
                ubx = lbxubx['ub'], \
                lbg = self.casadi_lbg, \
                ubg = self.casadi_ubg, \
                p = casadi.vertcat(p_value, casadi.DM(initPenaltyWeight), x_value[self.y_indices]),
                lam_x0 = previousResult['lam_x'],
                lam_g0 = previousResult['lam_g']
            )

        solver_stats = self.casadi_nlpsol.stats()
        assert solver_stats['success']
        return {'result': result, 'solver_stats': solver_stats}

    def solver_loop(self, x_value, p_value, **kwargs):
        initPenaltyWeight = kwargs.get('initPenaltyWeight', 100.0)
        beta = kwargs.get('beta', 0.1)
        penaltyWeightReductionSteps = kwargs.get('penaltyWeightReductionSteps', 5)
        solveOriginalProblem = kwargs.get('solveOriginalProblem', True)

        # Solve the problem repeatedly while reducing the initial guess penalty
        status = [{'result': None}]
        for i in range(penaltyWeightReductionSteps):
            status.append(self.solver_step(x_value, p_value, initPenaltyWeight, status[-1]['result']))
            initPenaltyWeight *= beta

        if solveOriginalProblem:
            # Final step without initial guess penalty, to solve the original problem
            status.append(self.solver_step(x_value, p_value, 0.0, status[-1]['result']))

        return status