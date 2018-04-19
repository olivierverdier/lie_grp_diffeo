import lie_group_diffeo as lgd
import odl
import numpy as np

def get_regularization(space, deform_action, regularizer='determinant'):
    # Define what regularizer to use
    lie_grp = deform_action.lie_group
    if regularizer == 'image':
        # Create set of all points in space
        W = space.tangent_bundle
        w = W.element(np.reshape(space.points().T, W.shape))

        # Create regularizing functional
        regularizer = 0.01 * odl.solvers.L2NormSquared(W).translated(w)

        # Create action
        regularizer_action = lgd.ProductSpaceAction(deform_action, len(W))
    elif regularizer == 'point':
        W = odl.ProductSpace(odl.rn(space.ndim), 3)
        w = W.element([[0, 0],
                    [0, 1],
                    [1, 0]])

        # Create regularizing functional
        regularizer = 0.01 * odl.solvers.L2NormSquared(W).translated(w)

        # Create action
        # if lie_grp_type == 'affine' or lie_grp_type == 'rigid':
        if isinstance(lie_grp, lgd.AffineGroup) or isinstance(lie_grp, lgd.EuclideanGroup):
            point_action = lgd.MatrixVectorAffineAction(lie_grp, W[0])
        else:
            point_action = lgd.MatrixVectorAction(lie_grp, W[0])
        regularizer_action = lgd.ProductSpaceAction(point_action, len(W))
    elif regularizer == 'determinant':
        W = odl.rn(1)
        w = W.element([1])

        # Create regularizing functional
        regularizer = 0.2 * odl.solvers.L2NormSquared(W).translated(w)

        # Create action
        regularizer_action = lgd.MatrixDeterminantAction(lie_grp, W)
    else:
        assert False
    return w, regularizer, regularizer_action


import pytest

@pytest.fixture(params=['image', 'point', 'determinant'])
def regularization_param(request):
    return request.param
