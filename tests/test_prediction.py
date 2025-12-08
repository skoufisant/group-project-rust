import prediction


def test_make_prediction_simple():
    """A simple test to check if make_prediction returns a float.

    This test must be modified as per the actual model used.

    """
    result = prediction.make_prediction(
        tenure=2,
        MonthlyCharges=12.3,
        TechSupport_yes=0,
        PhoneService_yes=1,
        Contract_one_year=0,
        Contract_two_year=0,
        InternetService_fiber_optic=0,
        OnlineSecurity_yes=0
    )
    assert isinstance(result, float)
