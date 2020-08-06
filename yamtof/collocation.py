
# LGL quadrature constants based on https://doi.org/10.21914/anziamj.v47i0.1033

n_nodes = 5
nodes = [0.0, 0.172673164646011, 0.5, 0.827326835353989, 1.0]
integration_matrix = [
    [ 0.0677284321861569,  0.119744769343412,  -0.0217357218665581,   0.0106358242254155, -0.00370013924241453],
    [           0.040625,  0.303184183323043,    0.177777777777778,  -0.0309619611008206,             0.009375],
    [ 0.0537001392424145,  0.261586397996807,    0.377291277422114,    0.152477452878811,  -0.0177284321861569],
    [               0.05,  0.272222222222222,    0.355555555555556,    0.272222222222222,                 0.05]
]

integration_weights = \
    [               0.05,  0.272222222222222,    0.355555555555556,    0.272222222222222,                 0.05]

interpolation_weights = [14.0, -32.6666666666667, 37.3333333333333, -32.6666666666667, 14.0]

if __name__ == '__main__':
    import numpy as np
    M = np.matrix(integration_matrix)
    n = np.array(np.matrix(nodes).T)
    n1 = n[1:,:]

    # Check that polynomials are integrated correctly.
    # The quadrature must integrate low order polynomials exactly.
    for i in range(5):
        e = (i+1) * M@(n**i) - (n1**(i+1))
        print(e)
        assert np.max(np.abs(e)) < 1e-14
