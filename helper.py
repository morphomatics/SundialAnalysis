import numpy as np
from morphomatics.manifold.util import align


def generalized_procrustes(surf):
    """ Generalized Procrustes analysis.
    :arg surf: list of surfaces to be aligned. The meshes must be in correspondence.
    """
    ref_v = np.copy(surf[0].v)
    old_ref_v = np.copy(ref_v + 1)

    n_steps = 0
    # do until convergence
    while (np.linalg.norm(ref_v - old_ref_v) > 1e-11 and 1000 > n_steps):
        n_steps = n_steps + 1
        old_ref_v = np.copy(ref_v)
        # align meshes to reference
        for i, s in enumerate(surf):
            s.v = np.array(align(s.v, ref_v))

        # compute new reference
        ref_v = np.mean(np.array([s.v for s in surf]), axis=0)
