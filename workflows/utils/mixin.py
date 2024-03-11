import luigi
from luigi.parameter import ParameterVisibility


class TaskListParameter(luigi.Parameter):
    # https://luigi.readthedocs.io/en/stable/_modules/luigi/parameter.html#TaskParameter

    def _warn_on_wrong_param_type(self, param_name, param_value):
        """Don't warn on dynamic_requires parameter."""
        pass


class DynamicRequiresMixin:
    dynamic_requires = TaskListParameter(
        default=[], visibility=ParameterVisibility.HIDDEN
    )

    def requires(self):
        return self.dynamic_requires
