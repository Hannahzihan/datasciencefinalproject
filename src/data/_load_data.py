from ucimlrepo import fetch_ucirepo 

def load_data():
    """Load and prepare the In-Vehicle Coupon Recommendation dataset from UCI repository.

    This function fetches the dataset from the UCI Machine Learning Repository using its unique ID (603) 
    and separates it into features (X) and targets (y) as pandas DataFrames. Additionally, it prints metadata 
    and variable information for reference.

    Source:
        In-Vehicle Coupon Recommendation. 2017. UCI Machine Learning Repository. 
        https://doi.org/10.24432/C5GS4P

    Returns:
        tuple: A tuple containing two pandas DataFrames (X, y), where:
            - X: Features of the dataset
            - y: Target labels of the dataset

    Notes:
        The dataset contains information relevant to the prediction of coupon acceptance 
        within vehicles, offering insights into user behavior.
    """    
    # fetch dataset 
    in_vehicle_coupon_recommendation = fetch_ucirepo(id=603) 
    
    # data (as pandas dataframes) 
    X = in_vehicle_coupon_recommendation.data.features 
    y = in_vehicle_coupon_recommendation.data.targets 
    
    # metadata 
    print("Metadata:")
    print(in_vehicle_coupon_recommendation.metadata) 

    return X, y

