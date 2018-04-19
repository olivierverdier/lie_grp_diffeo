import lie_group_diffeo as lgd
import odl
import numpy as np

from lie_group_diffeo.utils.deformation import get_deformation_action, deformation_param

from lie_group_diffeo.utils.regularization_1d import get_regularization_action, regularization_param



def test(deformation_param, regularization_param):
    space = odl.uniform_discr(-1, 1, 1000, interp='nearest')

    deform_action = get_deformation_action(space, deformation_param)
    w, regularizer, regularizer_action = get_regularization_action(space, deform_action, regularization_param)

    # Define template
    template = space.element(lambda x: np.exp(-x**2 / 0.2**2))
    target = space.element(lambda x: np.exp(-(x-0.1)**2 / 0.3**2))

    # Define data matching functional
    data_matching = odl.solvers.L2NormSquared(space).translated(target)



    # Initial guess
    g = deform_action.lie_group.identity

    # Combine action and functional into single object.
    action = lgd.ProductSpaceAction(deform_action, regularizer_action)
    # x = action.domain.element([template, np.broadcast_to(w, action.domain[1].shape)]).copy()
    f = odl.solvers.SeparableSum(data_matching, regularizer)
    x = f.domain.element([template, w]).copy()

    # Show some results, reuse the plot
    # fig = template.show()
    # target.show(fig=fig)

    # Create callback that displays the current iterate and prints the function
    # value
    # callback = odl.solvers.CallbackShow(lie_grp_type, step=10,
    #                                     indices=0)
    # callback &= odl.solvers.CallbackPrint(f)

    # Solve via gradient flow
    lgd.gradient_flow_solver(x, f, g, action,
                            niter=1, line_search=0.2)
