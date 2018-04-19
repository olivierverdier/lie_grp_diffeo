
import lie_group_diffeo as lgd
import odl
import numpy as np

def test_identity():
    space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='nearest')
    coord_space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='linear').tangent_bundle

    lie_grp = lgd.diff_group.Diff(space, coord_space=coord_space)

    g = lie_grp.identity

def test_prod():
    space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='nearest')
    coord_space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='linear').tangent_bundle
