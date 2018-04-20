import odl

class Solver:
    def __init__(self, template, f, action, rate=1.):
        self.template = template
        self.f = f
        self.algebra = action.lie_group.associated_algebra
        self.g0 = action.lie_group.identity
        self.Ainv = odl.IdentityOperator(self.algebra)
        self.action = action
        self.rate = rate
        self.line_search = odl.solvers.ConstantLineSearch(rate)

    def starting_guess(self):
        return self.g0

    def loss(self, g):
        return self.f(self.action.action(g)(self.template))

    def step(self, g):
        x = self.action.action(g)(self.template)
        fgrad_x = self.f.gradient(x)
        u = self.Ainv(self.action.momentum_map(x, fgrad_x))

        # direction = -self.action.inf_action(u)(x)
        # dir_derivative = fgrad_x.inner(direction)
#        steplen = self.line_search(x, direction, dir_derivative)
        steplen = self.rate

        dg = self.algebra.exp(-steplen * u)
        g_ = g.compose(dg)  # wrong direction?

        return g_

class Momentum(Solver):
    def damp(self, t):
        return 3/(t+1)

    def past(self, t, g, mom):
        return g

    def starting_guess(self):
        g0 = super().starting_guess()
        mu0 = self.algebra.zero()
        return 0, g0, mu0

    def step(self, t, g, mom):
        g__ = self.past(t, g, mom)
        x = self.action.action(g__)(self.template)
        dmom = self.action.momentum_map(x, self.f.gradient(x))
        mom_ = mom + self.rate*(-dmom - self.damp(t)*mom)
        vel = self.Ainv(mom_)
        dg = self.algebra.exp(self.rate*vel)
        g_ = g.compose(dg)
        t_ = t + self.rate
        return t_, g_, mom_


class Accelerated(Momentum):
    def past(self, t, g, mom):
        dg = self.algebra.exp(-self.rate*(1 - self.damp(t))*self.Ainv(mom))
        res = g.compose(dg)
        return res
