import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Minimum at: x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0


class Constraint(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('g', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['g'] = x + y


if __name__ == "__main__":

    model = om.Group()
    model.add_subsystem('parab_comp', Paraboloid(), promotes_inputs=['x', 'y'], promotes_outputs=['f_xy'])
    model.add_subsystem('const', Constraint(), promotes_inputs=['x', 'y'])

    # or more quickly
    # model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

    prob = om.Problem(model)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'

    # to add the design var and obj
    prob.model.add_design_var('x', lower=-50, upper=50)
    prob.model.add_design_var('y', lower=-50, upper=50)
    prob.model.add_objective('f_xy')

    # to add the constraint to the model
    prob.model.add_constraint('const.g', lower=0, upper=10.)

    prob.setup()
    prob.set_val('x', -3.0)
    prob.set_val('y', 4.0)
    prob.run_driver()