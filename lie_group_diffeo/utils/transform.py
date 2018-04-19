import odl
import numpy as np

def get_transform(space, transform_type='rotate'):
    # Select deformation type of the target
    if transform_type == 'affine':
        transform = odl.deform.LinDeformFixedDisp(
            space.tangent_bundle.element([lambda x: x[0] * 0.3 + x[1] * 0.2,
                                        lambda x: x[0] * 0.1 + x[1] * 0.3]))
    elif transform_type == 'rotate':
        theta = 0.2
        transform = odl.deform.LinDeformFixedDisp(
            space.tangent_bundle.element([lambda x: (np.cos(theta) - 1) * x[0] + np.sin(theta) * x[1],
                                        lambda x: -np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1]]))
    else:
        assert False
    return transform


import pytest

@pytest.fixture(params=['affine', 'rotate'])
def transform_param(request):
    return request.param
