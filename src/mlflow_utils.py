from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from joblib import dump,load
import os
from sklearn.metrics import r2_score, mean_absolute_error

from src.logger import logger
import models

models_path = models.path
if not os.path.exists(models_path):
    raise ValueError(f'Does not exist:{models_path}')
path = os.path.join(models_path,'mlruns')
tracking_uri = f'file:///{os.path.abspath(path)}'
mlflow.set_tracking_uri(tracking_uri)


def save_model(model: Any, model_name: str, model_dir:str=None) -> str:
    """
    Saves model in pickle format

    Args:
        model: Model binary
        model_name: Name of model
        model_dir: Directory to save model in

    Returns:
        Output path of model
    """

    if model_dir is None:
        model_dir = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'models')

    model_file_name = f'{model_name}.pickle'
    output_path = os.path.join(model_dir, model_file_name)
    logger.info('Model saved to: {}'.format(output_path))
    dump(model, output_path)

    return output_path

def load_model(run):

    model_name = run.data.params['model_name']
    model_file_name = f'{model_name}.pickle'
    path = run.info.artifact_uri.replace(r'file:///','')
    model_path = os.path.join(path,model_file_name)

    return load(model_path)



def log_mlflow(run_params: Dict, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Logs result of model training and validation to mlflow

    Args:
        run_params: Dictionary containing parameters of run.
                    Expects keys for 'experiment'
        model: Model binary
        model_name: Name of model
        y_test: Array of true y values
        X_test: Array of test features

    Returns:
        None
    """
    model_name = run_params['model_name']
    mlflow.set_experiment(run_params['experiment'])
    model_path = save_model(model, model_name)

    y_pred = model.predict(X_test)
    
    plot_dir='plots'
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    test_plot_path = plot_test(y_pred=y_pred, y_test=y_test, plot_dir=plot_dir, model_name=model_name)
    metrics = evaluate(y_pred=y_pred, y_test=y_test)

    with mlflow.start_run(run_name=run_params['run_name']) as run:
        mlflow.log_params(run_params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(test_plot_path)

    return run

def evaluate(y_pred, y_test):
    metrics = {
        'r2_score' : r2_score(y_true=y_test, y_pred=y_pred),
        'mean_absolute_error'  : mean_absolute_error(y_true=y_test, y_pred=y_pred)
    }
    return metrics

def plot_test(y_pred, y_test, model_name:str, plot_dir:str=None):

    fig,ax=plt.subplots()
    y_test.plot(ax=ax)
    ax.plot(y_test.index, y_pred)

    # Save figure
    if plot_dir:
        output_path = '{}/plot_test_{}.png'.format(plot_dir, model_name)
        
        plt.savefig(output_path)
        logger.info('Precision-recall curve saved to: {}'.format(output_path))
        return output_path