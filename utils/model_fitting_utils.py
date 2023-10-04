import pandas as pd
import xgboost as xgb
from typing import Dict
from IPython.display import display
from utils.model_utils import evaluate_model
from sklearn.ensemble import ExtraTreesRegressor


initial_params = {
    'extra_trees': {
        'n_estimators': 100,
        'max_depth': None,
        'criterion': 'squared_error',
        'n_jobs': -1,
        'random_state': 42
    },
    'xgboost': {
        "sampling_method": "gradient_based",
        'objective': 'reg:squarederror',
        'learning_rate': 0.23,
        'tree_method': 'hist',
        'booster': 'gbtree',
        'device': 'cuda',
        'max_depth': 11,
        "subsample": 1,
        "trees": 5000,
        "max_bin":500,
        "seed": 42
    },
    'pytorch_nn': {
        'input_size': 8,
        'hidden_size': 10,
        'num_classes': 1,
        'num_epochs': 5,
        'batch_size': 100,
        'learning_rate': 0.001
    }
}

class ModelFitting:

    def __init__(
        self,
        var: str,
        split: Dict[str, pd.DataFrame],
        model_name: str,
        **model_params
    ):
        self.var = var
        self.split = split
        self.model_name = model_name
        
        if self.model_name == 'extra_trees':
            self.model_params = {**initial_params['extra_trees'], **model_params}

        elif self.model_name == 'xgboost':
            self.model_params = {**initial_params['xgboost'], **model_params}
        
        elif self.model_name == 'pytorch_nn':
            self.model_params = {**initial_params['pytorch_nn'], **model_params}

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')

    
    def fit_model(self) -> None:
        """
        A function that fits the model to the training data.
        """

        if self.model_name == 'extra_trees':

            self.model = ExtraTreesRegressor(**self.model_params)
            X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
            self.model.fit(X_train, y_train)
            
        elif self.model_name == 'xgboost':

            dtrain=xgb.DMatrix(self.split['X_temp'][self.var], label=self.split['y_train'])
            n_trees = self.model_params["trees"]
            boosting_params = self.model_params.copy()
            boosting_params.pop("trees")

            self.model = xgb.train(
                params=boosting_params,
                dtrain=dtrain,
                num_boost_round=n_trees
            )

        elif self.model_name == 'pytorch_nn':

            pass

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')




    def make_predictions(self) -> Dict[str, pd.DataFrame]:
        """
        A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.
        """

        if self.model_name == 'extra_trees':

            self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
            y_pred_test_df = pd.DataFrame(
                data = self.y_pred_test,
                columns = ['snow_depth_pred']
            )

            self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
            y_pred_train_df = pd.DataFrame(
                data = self.y_pred_train,
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }
            

        elif self.model_name == 'xgboost':

            self.y_pred_test = self.model.predict(
                xgb.DMatrix(self.split['X_test'][self.var])
            )
            y_pred_test_df = pd.DataFrame(
                data = self.y_pred_test,
                columns = ['snow_depth_pred']
            )

            self.y_pred_train = self.model.predict(
                xgb.DMatrix(self.split['X_temp'][self.var])
            )
            y_pred_train_df = pd.DataFrame(
                data = self.y_pred_train,
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }

        elif self.model_name == 'pytorch_nn':

            pass

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')
        

    def evaluate_model(self) -> pd.DataFrame:

        """
        A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.
        """

        if self.model_name == 'extra_trees':

            train_eval=evaluate_model(
                y_true=self.split['y_temp'][self.var],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=evaluate_model(
                y_true=self.split['y_test'][self.var],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'xgboost':

            train_eval=evaluate_model(
                y_true=self.split['y_temp'][self.var],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=evaluate_model(
                y_true=self.split['y_test'][self.var],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'pytorch_nn':

            pass

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')
        
    
    def get_importance(self) -> pd.DataFrame:

        if self.model_name == 'extra_trees':

            feature_importance = pd.DataFrame(
                data = {
                    'feature': self.split['X_temp'][self.var].columns,
                    'importance': self.model.feature_importances_
                }
            ).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        elif self.model_name == 'xgboost':

            feature_importance = pd.DataFrame(
                data = {
                    'feature': self.split['X_temp'].columns,
                    'importance': self.model.get_score(importance_type='gain').values()
                }
            ).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        elif self.model_name == 'pytorch_nn':

            pass

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')
        