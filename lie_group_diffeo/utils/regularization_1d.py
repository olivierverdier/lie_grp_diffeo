import lie_group_diffeo as lgd
import odl


def get_regularization_action(space, deform_action, regularizer='image'):
    # Define what regularizer to use
    lie_grp = deform_action.lie_group
    if regularizer == 'image':
        # Create set of all points in space
        W = space.tangent_bundle
        w = W.element(space.points().T)

        # Create regularizing functional
        regularizer = 0.01 * odl.solvers.L2NormSquared(W).translated(w)

        # Create action
        regularizer_action = lgd.ProductSpaceAction(deform_action, len(W))
    elif regularizer == 'point':
        W = odl.ProductSpace(odl.rn(space.ndim), 2)
        w = W.element([[0], [1]])

        # Create regularizing functional
        regularizer = 0.01 * odl.solvers.L2NormSquared(W).translated(w)

        # Create action
        if isinstance(lie_grp, lgd.AffineGroup) or isinstance(lie_grp, lgd.EuclideanGroup):
            point_action = lgd.MatrixVectorAffineAction(lie_grp, W[0])
        else:
            point_action = lgd.MatrixVectorAction(lie_grp, W[0])
        regularizer_action = lgd.ProductSpaceAction(point_action, W.size)
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
