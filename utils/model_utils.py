import numpy as np
import pandas as pd
from typing import Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set the seed for reproducibility
seed = 10



def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    """
    A function that splits the data into training (80%), testing (10%) and tuning (10%) sets.

    Parameters:
    -----------
    df : pandas DataFrame
        A pandas DataFrame containing the data to split.

    Returns:
    --------
    A dictionary containing the training, testing and tuning sets.
    """

    X,y = df.drop('snow_depth', axis=1), df['snow_depth']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1/9, random_state=seed
    )


    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'X_temp': X_temp,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
        'y_temp': y_temp
    }


# let's create a function that will help us evaluate the results of the model

def evaluate_model(
    y_true: Union[pd.Series, list], 
    y_pred: Union[pd.Series, list], 
    model_name: str
) -> pd.DataFrame:

    """
    A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.

    Parameters:
    -----------
    y_true : pandas Series or list
        A pandas Series or list containing the true values of the target variable.

    y_pred : pandas Series or list
        A pandas Series or list containing the predicted values of the target variable.
    
    model_name : str
        A string representing the name of the model.

    Returns:
    --------
    A pandas DataFrame containing the RMSE, MBE and R2 metrics for the model.
    """

    RMSE = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    MBE  = np.mean(y_pred- y_true)
    RSQ  = r2_score(y_true=y_true, y_pred=y_pred)
    
    score_df = pd.DataFrame({
        model_name: [RMSE, MBE, RSQ]
    }, index = ['RMSE', 'MBE', 'RSQ'])
    
    return score_df
