from dagshub.common.util import lazy_load, multi_urljoin
from dagshub.common.api import UserAPI
from dagshub.auth import get_token
from dagshub.common import config

from typing import TYPE_CHECKING, Optional
import importlib
import os

if TYPE_CHECKING:
    import mlflow
else:
    mlflow = lazy_load("mlflow")


def get_mlflow_model(repo: str, name: str, version: Optional[str] = "latest", host: Optional[str] = None):
    """
    Load an MLflow registered model from the specified DagsHub repository.
    Call `model.predict(<data>)` on the loaded model to infer on your data.

    Example::

        model = dagshub.mlflow.get_mlflow_model('jinensetpal/COCO_1K', 'yolov8-seg', 'latest', 'https://dagshub.com')
        model.predict("https://www.dagshub.com/jinensetpal/COCO_1K/raw/68a54c3bdc84582af3e42fb6f03507146f176378/data/images/train/000000000009.jpg")

    Args:
        repo: repository to extract the model from
        name: name of the model in the repository's MLflow registry.
        host: address of the DagsHub instance with the repo to load the model from.
            Set it if the model is hosted on a different DagsHub instance than the datasource.
        version: version of the model in the mlflow registry.
    """
    host = host or config.host

    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(multi_urljoin(host, f"{repo}.mlflow"))

    token = get_token(host=host)
    os.environ["MLFLOW_TRACKING_USERNAME"] = UserAPI.get_user_from_token(token, host=host).username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    model_uri = f"models:/{name}/{version}"

    try:
        loader_module = mlflow.models.get_model_info(model_uri).flavors["python_function"]["loader_module"]
        loader_module_elems = loader_module.split(".")
        if loader_module_elems[-1] == "model":
            loader_module_elems.pop()
        loader_module = ".".join(loader_module_elems)
        loader = mlflow.pyfunc if "pyfunc" in loader_module_elems else importlib.import_module(loader_module)
        model = loader.load_model(model_uri)
    finally:
        mlflow.set_tracking_uri(prev_uri)

    if "torch" in loader_module:
        model.predict = model.__call__

    return model
