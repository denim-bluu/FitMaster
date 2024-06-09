from .concrete import LinearForm, ExponentialForm, LogarithmicForm
from .interface import FunctionalFormStrategy


class FunctionalFormFactory:
    """
    A factory class for creating instances of FunctionalFormStrategy based on the given functional form name.

    Attributes:
        functional_forms (dict[str, FunctionalFormStrategy]): A dictionary mapping functional form names to their corresponding strategies.
    """

    def __init__(self) -> None:
        self.functional_forms: dict[str, FunctionalFormStrategy] = {
            "linear": LinearForm(),
            "exponential": ExponentialForm(),
            "logarithmic": LogarithmicForm(),
        }

    def get_functional_form(self, functional_form: str) -> FunctionalFormStrategy:
        """
        Returns the FunctionalFormStrategy instance based on the given functional form name.

        Args:
            functional_form (str): The name of the functional form.

        Returns:
            FunctionalFormStrategy: The instance of the FunctionalFormStrategy corresponding to the given functional form name.

        Raises:
            ValueError: If the given functional form name is not found in the functional_forms dictionary.

        """
        if functional_form not in self.functional_forms:
            raise ValueError(f"Functional form '{functional_form}' not found.")
        return self.functional_forms[functional_form]
