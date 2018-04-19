import lie_group_diffeo as lgd

def get_deformation_action(space, lie_grp_type='affine'):
    # Define the lie group to use.
    if lie_grp_type == 'gln':
        lie_grp = lgd.GLn(space.ndim)
        deform_action = lgd.MatrixImageAction(lie_grp, space)
    elif lie_grp_type == 'son':
        lie_grp = lgd.SOn(space.ndim)
        deform_action = lgd.MatrixImageAction(lie_grp, space)
    elif lie_grp_type == 'sln':
        lie_grp = lgd.SLn(space.ndim)
        deform_action = lgd.MatrixImageAction(lie_grp, space)
    elif lie_grp_type == 'affine':
        lie_grp = lgd.AffineGroup(space.ndim)
        deform_action = lgd.MatrixImageAffineAction(lie_grp, space)
    elif lie_grp_type == 'rigid':
        lie_grp = lgd.EuclideanGroup(space.ndim)
        deform_action = lgd.MatrixImageAffineAction(lie_grp, space)
    else:
        assert False
    return deform_action

import pytest

@pytest.fixture(params=['gln', 'son', 'sln', 'affine', 'rigid'])
def deformation_param(request):
    return request.param


def get_deform_action(space, lie_grp, action_type='geometric'):
    geometric_deform_action = lgd.GeometricDeformationAction(lie_grp, space)
    scale_action = lgd.JacobianDeterminantScalingAction(lie_grp, space)
    if action_type == 'mass_preserving':
        deform_action = lgd.ComposedAction(geometric_deform_action, scale_action)
    elif action_type == 'geometric':
        deform_action = geometric_deform_action
    else:
        assert False
    return geometric_deform_action, deform_action

@pytest.fixture(params=['mass_preserving', 'geometric'])
def action_param(request):
    return request.param
