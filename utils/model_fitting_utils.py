import pandas as pd
from typing import Dict
from IPython.display import display


from utils.model_utils import evaluate_model
from utils.pytorch_model import RegressionNN
from utils.pytorch_training import train, predict
from utils.pytorch_dataset import create_dataset_for_dnn

import xgboost as xgb
from sklearn import set_config
from sklearn.ensemble import ExtraTreesRegressor

set_config(
    transform_output="pandas"
)

import torch
import torch.nn as nn
import torch.optim as optim

xgb_device = ("cuda" if torch.cuda.is_available() else "cpu")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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
        "min_child_weight": 30,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'booster': 'gbtree',
        'device': xgb_device,
        'max_depth': 0,
        "subsample": 1,
        "max_bin":5096,
        "trees": 1000,
        "seed": 42
    },
    'pytorch_nn': {
        'hidden_size1': 2048,
        'hidden_size2': 1500,
        'hidden_size3': 1000,
        'num_epochs': 15,
        'batch_size': 128,
        'learning_rate': 0.0001,
        'verbose': True
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
            raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn.')

    
    def fit_model(self) -> None:
        """
        A function that fits the model to the training data.
        """

        if self.model_name == 'extra_trees':

            self.model = ExtraTreesRegressor(**self.model_params)
            X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
            self.model.fit(X_train, y_train)
            
        elif self.model_name == 'xgboost':

            dtrain=xgb.DMatrix(self.split['X_temp'][self.var], label=self.split['y_temp'])
            n_trees = self.model_params["trees"]
            boosting_params = self.model_params.copy()
            boosting_params.pop("trees")

            self.model = xgb.train(
                params=boosting_params,
                dtrain=dtrain,
                num_boost_round=n_trees
            )

        elif self.model_name == 'pytorch_nn':

            self.loader = create_dataset_for_dnn(
                split=self.split, 
                columns_of_interest=self.var, 
                batch_size=self.model_params['batch_size']
            )

            input_size = self.loader['train_dataloader'].dataset.features.shape[1]
            hidden_size1 = self.model_params['hidden_size1']
            hidden_size2 = self.model_params['hidden_size2']
            hidden_size3 = self.model_params['hidden_size3']

            self.model = RegressionNN(
                input_size=input_size,
                hidden_size1=hidden_size1,
                hidden_size2=hidden_size2,
                hidden_size3=hidden_size3
            )

            optimizer = optim.Adam(self.model.parameters(), lr= self.model_params['learning_rate'])
            criterion = nn.MSELoss()

            self.history = train(
                model=self.model,
                train_loader=self.loader['train_dataloader'],
                val_loader=self.loader['val_dataloader'],
                epochs= self.model_params['num_epochs'],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                metric='mae',
                verbose=self.model_params['verbose']
            )

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

            self.predictions_test = predict(
                model=self.model,
                test_loader=self.loader['test_dataloader'],
                device=device
            )

            self.predictions_train = predict(
                model=self.model,
                test_loader=self.loader['train_dataloader'],
                device=device
            )

            y_pred_test_df = pd.DataFrame(
                data = self.predictions_test['predictions'],
                columns = ['snow_depth_pred']
            )

            y_pred_train_df = pd.DataFrame(
                data = self.predictions_train['predictions'],
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }

        else:
            raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn.')
        

    def evaluate_model(self) -> pd.DataFrame:

        """
        A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.
        """

        if self.model_name == 'extra_trees':

            train_eval=evaluate_model(
                y_true=self.split['y_temp'],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=evaluate_model(
                y_true=self.split['y_test'],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'xgboost':

            train_eval=evaluate_model(
                y_true=self.split['y_temp'],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=evaluate_model(
                y_true=self.split['y_test'],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'pytorch_nn':

            train_eval=evaluate_model(
                y_true=self.predictions_train['labels'],
                y_pred=self.predictions_train['predictions'],
                model_name=self.model_name + '_train'
            )

            test_eval=evaluate_model(
                y_true=self.predictions_test['labels'],
                y_pred=self.predictions_test['predictions'],
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        else:
            raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn.')
    
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
                    'feature': self.split['X_temp'][self.var].columns,
                    'importance': self.model.get_score(importance_type='gain').values()
                }
            ).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        elif self.model_name == 'pytorch_nn':
            print('No feature importance for PyTorch NNs.')

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')