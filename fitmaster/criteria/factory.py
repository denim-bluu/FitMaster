from .concrete import AICCriterion, BICCriterion, RSquaredCriterion
from .interface import ModelSelectionCriterionStrategy


class ModelSelectionCriterionFactory:
    """
    Factory class for creating model selection criterion strategies.

    This factory class provides a way to create different model selection criterion strategies
    based on the given criterion name.

    Attributes:
        criterions (dict[str, ModelSelectionCriterionStrategy]): A dictionary mapping criterion names to
            corresponding model selection criterion strategies.
    """

    def __init__(self) -> None:
        self.criterions: dict[str, ModelSelectionCriterionStrategy] = {
            "aic": AICCriterion(),
            "bic": BICCriterion(),
            "r_squared": RSquaredCriterion(),
        }

    def get_criteria(self, criterion: str) -> ModelSelectionCriterionStrategy:
        """
        Returns the model selection criterion strategy based on the given criterion name.

        Args:
            criterion (str): The name of the criterion.

        Returns:
            ModelSelectionCriterionStrategy: The model selection criterion strategy.

        Raises:
            ValueError: If the given criterion name is not found in the factory.
        """
        if criterion not in self.criterions:
            raise ValueError(f"Criterion '{criterion}' not found.")
        return self.criterions[criterion]
