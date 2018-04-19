import lie_group_diffeo as lgd
import odl
import numpy as np

from lie_group_diffeo.utils.transform import get_transform, transform_param

from lie_group_diffeo.utils.deformation import get_deformation_action, deformation_param

from lie_group_diffeo.utils.regularization_2d import get_regularization, regularization_param


def test(deformation_param, regularization_param, transform_param):

    # Select space and interpolation
    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300], interp='nearest')

    deform_action = get_deformation_action(space, deformation_param)
    w, regularizer, regularizer_action = get_regularization(space, deform_action, regularization_param)

    transform = get_transform(space, transform_param)


    # Create template and target
    template = odl.phantom.shepp_logan(space, modified=True)
    target = transform(template)

    # Define data matching functional
    data_matching = odl.solvers.L2NormSquared(space).translated(target)



    # Initial guess
    g = deform_action.lie_group.identity

    # Combine action and functional into single object.
    action = lgd.ProductSpaceAction(deform_action, regularizer_action)
    x = action.domain.element([template, w]).copy()
    f = odl.solvers.SeparableSum(data_matching, regularizer)

    # Show some results, reuse the plot
    # template.show('template')
    # target.show('target')

    # Create callback that displays the current iterate and prints the function
    # value
    # callback = odl.solvers.CallbackShow(lie_grp_type, step=10, indices=0)
    # callback &= odl.solvers.CallbackPrint(f)

    # Solve via gradient flow
    lgd.gradient_flow_solver(x, f, g, action,
                            niter=1, line_search=0.002)
