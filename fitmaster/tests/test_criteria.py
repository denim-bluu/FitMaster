from fitmaster.criteria.concrete import AICCriterion, BICCriterion, RSquaredCriterion
from numpy.testing import assert_almost_equal


def test_aic_criterion(y_data, non_matching_y_pred):
    aic = AICCriterion()
    num_params = 2
    assert_almost_equal(
        aic.evaluate(y_data, non_matching_y_pred, num_params), -9.81551055796427
    )


def test_bic_criterion(y_data, non_matching_y_pred):
    bic = BICCriterion()
    num_params = 2
    assert_almost_equal(
        bic.evaluate(y_data, non_matching_y_pred, num_params), -11.61828598062805
    )


def test_r_squared_criterion(y_data, matching_y_pred):
    r_squared = RSquaredCriterion()
    num_params = 2
    assert_almost_equal(r_squared.evaluate(y_data, matching_y_pred, num_params), 1.0)
