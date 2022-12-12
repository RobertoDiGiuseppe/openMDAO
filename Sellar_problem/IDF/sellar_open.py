import numpy as np
import openmdao.api as om


class SellarDis1(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """
    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))
        # Local Design Variable
        self.add_input('x', val=0.)
        # Coupling parameter
        self.add_input('y2', val=1.0)
        # Coupling output
        self.add_output('y1', val=1.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x1 = inputs['x']
        y2 = inputs['y2']

        outputs['y1'] = z1**2 + z2 + x1 - 0.2*y2


# This code defines the second discipline
class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs['y2'] = y1**.5 + z1 + z2


# This code defines the overall group by promoting all the variables
class SellarMDA(om.Group):
    """
    Group containing the Sellar MDA.
    """

    def setup(self):
        self.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z', 'y2'],
                            promotes_outputs=['y1'])
        self.add_subsystem('d2', SellarDis2(), promotes_inputs=['z', 'y1'],
                            promotes_outputs=['y2'])

        self.set_input_defaults('x', 0.)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])


# Now let's define the optimization problem
prob = om.Problem()
prob.model = SellarMDA()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['maxiter'] = 100
prob.driver.options['tol'] = 1e-8

prob.model.add_design_var('x', lower=0, upper=10)
prob.model.add_design_var('z', lower=0, upper=10)
prob.model.add_objective('obj')
prob.model.add_constraint('con1', upper=0)
prob.model.add_constraint('con2', upper=0)

# Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
prob.model.approx_totals()

# # Create a recorder variable in order to store the data for post-process
# recorder = om.SqliteRecorder('cases.sql')
# # Attach a recorder to the problem
# prob.add_recorder(recorder)

prob.setup()
prob.set_solver_print(level=0)

prob.run_driver()

# # The code that follows allows to visualize the results
# prob.record("after_run_driver")
# # Instantiate your CaseReader
# cr = om.CaseReader("cases.sql")
# # Isolate "problem" as your source
# driver_cases = cr.list_cases('problem', out_stream=None)
# # Get the first case from the recorder
# case = cr.get_case('after_run_driver')
# # list_outputs will list your model's outputs and return a list of them too
# print(case.list_outputs())
# # To visualize other variables that are not outputs
# design_vars = case.get_design_vars()
# print(design_vars['z'])