import datetime
import inspect
import logging
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import field, dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union, Tuple, Literal, Callable
import json
import os
import os.path

import dacite
import dagshub_annotation_converter.converters.yolo
import rich.progress
from dagshub_annotation_converter.formats.yolo import YoloContext
from dagshub_annotation_converter.formats.yolo.categories import Categories
from dagshub_annotation_converter.formats.yolo.common import ir_mapping
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase
from dagshub.mlflow import get_mlflow_model
from pydantic import ValidationError

from dagshub.common import config
from dagshub.common.util import lazy_load
from dagshub.common.analytics import send_analytics_event
from dagshub.common.download import download_files
from dagshub.common.helpers import sizeof_fmt, prompt_user, log_message
from dagshub.common.rich_util import get_rich_progress
from dagshub.data_engine.annotation import MetadataAnnotations
from dagshub.data_engine.annotation.voxel_conversion import (
    add_voxel_annotations,
    add_ls_annotations,
)
from dagshub.data_engine.client.models import DatasourceType, MetadataSelectFieldSchema
from dagshub.data_engine.model.datapoint import Datapoint, _get_blob, _generated_fields
from dagshub.data_engine.client.loaders.base import DagsHubDataset
from dagshub.data_engine.model.schema_util import dacite_config
from dagshub.data_engine.voxel_plugin_server.utils import set_voxel_envvars
from dagshub.data_engine.dtypes import MetadataFieldType

if TYPE_CHECKING:
    from dagshub.data_engine.model.datasource import Datasource
    import fiftyone as fo
    import dagshub.data_engine.voxel_plugin_server.server as plugin_server_module
    import datasets as hf_ds
    import tensorflow as tf
    import mlflow
    import mlflow.entities
else:
    plugin_server_module = lazy_load("dagshub.data_engine.voxel_plugin_server.server")
    fo = lazy_load("fiftyone")
    tf = lazy_load("tensorflow")
    hf_ds = lazy_load("datasets")
    mlflow = lazy_load("mlflow")

logger = logging.getLogger(__name__)

CustomPredictor = Callable[
    [
        List[str],
    ],
    List[Tuple[Any, Optional[float]]],
]


class VisualizeError(Exception):
    """:meta private:"""

    pass


def identity_func(x):
    return x


@dataclass
class QueryResult:
    """
    Result of executing a query on a :class:`.Datasource`.

    You can iterate over this object to get the :class:`datapoints <.Datapoint>`::

        res = ds.head()
        for dp in res:
            print(dp.path_in_repo)

    """

    _entries: List[Datapoint]
    datasource: "Datasource"
    fields: List[MetadataSelectFieldSchema]
    query_data_time: Optional[datetime.datetime] = None
    _datapoint_path_lookup: Dict[str, Datapoint] = field(init=False)

    def __post_init__(self):
        self._refresh_lookups()

    @property
    def entries(self):
        """
        list(Datapoint): Datapoints contained in this QueryResult
        """
        return self._entries

    @entries.setter
    def entries(self, value: List[Datapoint]):
        self._entries = value
        self._refresh_lookups()

    def _refresh_lookups(self):
        self._datapoint_path_lookup = {}
        for e in self.entries:
            self._datapoint_path_lookup[e.path] = e

    @property
    def dataframe(self):
        """
        Represent the contents of this QueryResult as a
        `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

        The created dataframe has a copy of the QueryResult's data.
        """
        import pandas as pd

        metadata_key_set = set()
        for e in self.entries:
            metadata_key_set.update(e.metadata.keys())

        metadata_keys = list(sorted(metadata_key_set))
        return pd.DataFrame.from_records([dp.to_dict(metadata_keys) for dp in self.entries])

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        """You can iterate over a QueryResult to get containing datapoints"""
        return self.entries.__iter__()

    def __repr__(self):
        return f"QueryResult of datasource {self.datasource.source.name} with {len(self.entries)} datapoint(s)"

    @staticmethod
    def from_gql_query(query_resp: Dict[str, Any], datasource: "Datasource") -> "QueryResult":
        raw_fields = query_resp.get("selectFields") or []
        fields = [dacite.from_dict(MetadataSelectFieldSchema, f, dacite_config) for f in raw_fields]
        # If no fields - get the default datasource ones
        if len(fields) == 0:
            fields = [MetadataSelectFieldSchema.from_metadata_field_schema(mfs) for mfs in datasource.fields]

        edges = query_resp.get("edges", [])
        if edges is None:
            edges = []
        datapoints = [Datapoint.from_gql_edge(edge, datasource, fields) for edge in edges]
        query_data_time = datetime.datetime.fromtimestamp(query_resp.get("queryDataTime"), tz=datetime.timezone.utc)

        return QueryResult(_entries=datapoints, datasource=datasource, fields=fields, query_data_time=query_data_time)

    def as_ml_dataset(self, flavor: str, **kwargs):
        """
        Convert the QueryResult into a dataset for a machine learning framework

        Args:
            flavor: Either of:

                - ``"torch"``: returns a \
                `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
                - ``"tensorflow"``: returns a \
                `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_


        Keyword Args:
            metadata_columns (list(str)): which fields to use in the dataset.
            strategy (str): Datapoint file loading strategy. Possible values:

                - ``"preload"`` - Download all datapoints before returning the dataset.
                - ``"background"`` - Start downloading the datapoints in the background. \
                    If an undownloaded datapoint is accessed, it gets downloaded.
                - ``"lazy"`` (default) - Download each datapoint as it is being accessed by the dataset.

            savedir (str|Path): Where to store the datapoint files. Default is :func:`datasource's default location \
                <dagshub.data_engine.model.datasource.Datasource.default_dataset_location>`
            processes (int): number of parallel processes to download the datapoints with. Default is 8.
            tensorizers: How to transform the datapoint file/metadata into tensors. Possible values:

                - ``"auto"`` - try to guess the tensorizers for every field.\
                    For files the tensorizer will be the determined by the first file's extension.
                - ``"image"`` | ``"audio"`` - tensorize all fields according to this type
                - List of tensorizers. First is the tensorizer for the datapoint file and then \
                    a tensorizer for each of the metadata fields. Tensorizer can either be strings\
                     ``"image"``, ``"audio"``, ``"numeric"``, or your own function that receives the\
                     metadata value of the field and turns it into a tensor.
        """
        flavor = flavor.lower()
        if flavor == "torch":
            from dagshub.data_engine.client.loaders.torch import PyTorchDataset

            return PyTorchDataset(self, **kwargs)
        elif flavor == "tensorflow":
            from dagshub.data_engine.client.loaders.tf import TensorFlowDataset

            ds_builder = TensorFlowDataset(self, **kwargs)
            ds = tf.data.Dataset.from_generator(ds_builder.generator, output_signature=ds_builder.signature)
            ds.__len__ = lambda: ds_builder.__len__()
            ds.__getitem__ = ds_builder.__getitem__
            ds.builder = ds_builder
            ds.type = "tensorflow"
            return ds
        else:
            raise ValueError("supported flavors are torch|tensorflow")

    def as_ml_dataloader(self, flavor, **kwargs):
        """
        Convert the QueryResult into a dataloader for a machine learning framework

        Args:
            flavor: Either of:

                - ``"torch"``: returns a \
                `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
                - ``"tensorflow"``: returns a \
                `tf.keras.utils.Sequence <https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence>`_

        Kwargs are the same as :func:`as_ml_dataset()`.
        """

        send_analytics_event(
            "Client_DataEngine_DataLoaderInitialized",
            repo=self.datasource.source.repoApi,
        )

        def keypairs(keys):
            return {key: kwargs[key] for key in keys}

        if type(flavor) is not str:
            if flavor.type == "torch":
                from dagshub.data_engine.client.loaders.torch import PyTorchDataLoader

                return PyTorchDataLoader(flavor, **kwargs)
            elif flavor.type == "tensorflow":
                from dagshub.data_engine.client.loaders.tf import TensorFlowDataLoader

                return TensorFlowDataLoader(flavor, **kwargs)

        kwargs["for_dataloader"] = True
        dataset_kwargs = set(list(inspect.signature(DagsHubDataset).parameters.keys())[1:])
        global_kwargs = set(kwargs.keys())
        flavor = flavor.lower() if type(flavor) is str else flavor
        if flavor == "torch":
            from dagshub.data_engine.client.loaders.torch import PyTorchDataLoader

            return PyTorchDataLoader(
                self.as_ml_dataset(flavor, **keypairs(global_kwargs.intersection(dataset_kwargs))),
                **keypairs(global_kwargs - dataset_kwargs),
            )
        elif flavor == "tensorflow":
            from dagshub.data_engine.client.loaders.tf import TensorFlowDataLoader

            return TensorFlowDataLoader(
                self.as_ml_dataset(
                    flavor,
                    **keypairs(global_kwargs.intersection(dataset_kwargs)),
                ),
                **keypairs(global_kwargs - dataset_kwargs),
            )
        else:
            raise ValueError("supported flavors are torch|tensorflow|<torch.utils.data.Dataset>|<tf.data.Dataset>")

    def as_hf_dataset(
        self, target_dir: Optional[Union[str, PathLike]] = None, download_datapoints=True, download_blobs=True
    ):
        """
        Loads this QueryResult as a HuggingFace dataset.

        The paths of the downloads are set to the local paths in the filesystem, so they can be used with
        a `cast_column() \
        <https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.cast_column>`_\
        function later.

        Args:
            target_dir: Where to download the datapoints. The metadata is still downloaded into the global cache.
            download_datapoints: If set to ``True`` (default), downloads the datapoint files and sets the path column\
                to the path of the datapoint in the filesystem
            download_blobs: If set to ``True`` (default), downloads all blob fields and sets the respective column\
                to the path of the file in the filesystem.
        """
        if download_blobs:
            # Download blobs as paths, so later a user can apply ds.cast_column on the blobs
            self.get_blob_fields(load_into_memory=False, path_format="str")

        df = self.dataframe

        if download_datapoints:
            # Do the same for the actual datapoint files, changing the path

            if target_dir is None:
                target_dir = self.datasource.default_dataset_location
            elif isinstance(target_dir, str):
                target_dir = Path(target_dir).absolute()
            new_paths = []
            self.download_files(target_dir=target_dir)
            for dp in df["path"]:
                new_paths.append(str(target_dir / self.datasource.source.source_prefix / dp))
            df["path"] = new_paths

        # Drop the generated fields
        for f in _generated_fields.keys():
            if f == "path":
                continue
            df.drop(f, axis=1, inplace=True)

        return hf_ds.Dataset.from_pandas(df)

    def __getitem__(self, item: Union[str, int, slice]):
        """
        Gets datapoint by its path (string) or by its index in the result (or slice)
        """
        if isinstance(item, str):
            return self._datapoint_path_lookup[item]
        elif isinstance(item, int):
            return self.entries[item]
        elif type(item) is slice:
            return QueryResult(_entries=self.entries[item], datasource=self.datasource, fields=self.fields)
        else:
            raise ValueError(
                f"Can't lookup datapoint using value {item} of type {type(item)}, "
                f"needs to be either int or str or slice"
            )

    def get_blob_fields(
        self,
        *fields: str,
        load_into_memory=False,
        cache_on_disk=True,
        num_proc: int = config.download_threads,
        path_format: Literal["str", "path"] = "path",
    ) -> "QueryResult":
        """
        Downloads data from blob fields.

        If ``load_into_memory`` is set to ``True``, then we additionally convert special fields to new types:

            - Annotation fields are converted to \
                :class:`MetadataAnnotations <dagshub.data_engine.annotation.metadata.MetadataAnnotations>`
            - Document fields are converted to strings

        Args:
            fields: list of binary fields to download blobs for. If empty, download all blob fields.
            load_into_memory: Whether to load the blobs into the datapoints, or just store them on disk


                If True: the datapoints' specified fields will contain the blob data

                If False: the datapoints' specified fields will contain Path objects to the file of the downloaded blob
            cache_on_disk: Whether to cache the blobs on disk or not (valid only if load_into_memory is set to True)
                Cache location is ``~/dagshub/datasets/<repo>/<datasource_id>/.metadata_blobs/``
            num_proc: number of download threads
            path_format: What way the paths to the file should be represented.
                ``path`` returns a Path object, and ``str`` returns a string of this path.
        """
        send_analytics_event("Client_DataEngine_downloadBlobs", repo=self.datasource.source.repoApi)
        if not load_into_memory:
            assert cache_on_disk

        # If no fields are specified, include all blob fields from self..fields
        if not fields:
            fields = tuple(
                [
                    f.name
                    for f in self.fields
                    if f.valueType == MetadataFieldType.BLOB and f.name not in self.document_fields
                ]
            )

        if len(fields) == 0:
            logger.warning("No blob fields loaded")
            return self

        cache_location = self.datasource.default_dataset_location / ".metadata_blobs"

        if cache_on_disk:
            cache_location.mkdir(parents=True, exist_ok=True)

        # Create a list of things to download: (datapoint, field, url, path_location)
        to_download: List[Tuple[Datapoint, str, str, Path]] = []
        for dp in self.entries:
            for fld in fields:
                field_value = dp.metadata.get(fld)
                # If field_value is a blob or a path, then ignore, means it's already been downloaded
                if not isinstance(field_value, str):
                    continue
                download_task = (dp, fld, dp.blob_url(field_value), dp.blob_cache_location / field_value)
                to_download.append(download_task)

        progress = get_rich_progress(rich.progress.MofNCompleteColumn())
        task = progress.add_task("Downloading blobs...", total=len(to_download))

        auth = self.datasource.source.repoApi.auth

        def _get_blob_fn(dp: Datapoint, field: str, url: str, blob_path: Path):
            blob_or_path = _get_blob(url, blob_path, auth, cache_on_disk, load_into_memory, path_format)
            if isinstance(blob_or_path, str) and path_format != "str":
                logger.warning(f"Error while downloading blob for field {field} in datapoint {dp.path}:{blob_or_path}")
            dp.metadata[field] = blob_or_path

        with progress:
            with ThreadPoolExecutor(max_workers=num_proc) as tp:
                futures = [tp.submit(_get_blob_fn, *args) for args in to_download]
                for f in as_completed(futures):
                    exc = f.exception()
                    if exc is not None:
                        logger.warning(f"Got exception {type(exc)} while downloading blob: {exc}")
                    progress.update(task, advance=1)

        self._convert_annotation_fields(*fields, load_into_memory=load_into_memory)

        # Convert any downloaded document fields
        document_fields = [f for f in fields if f in self.document_fields]
        if document_fields:
            for dp in self:
                for fld in document_fields:
                    if fld in dp.metadata:
                        # Override the load_into_memory flag, because we need the contents
                        if not load_into_memory:
                            dp.metadata[fld] = Path(dp.metadata[fld]).read_bytes()
                        dp.metadata[fld] = dp.metadata[fld].decode("utf-8")

        return self

    def _convert_annotation_fields(self, *fields, load_into_memory):
        # Convert any downloaded annotation column
        annotation_fields = [f for f in fields if f in self.annotation_fields]

        bad_annotations = defaultdict(list)

        if annotation_fields:
            # Convert them
            for dp in self:
                for fld in annotation_fields:
                    if fld in dp.metadata:
                        # Already loaded - skip
                        if isinstance(dp.metadata[fld], MetadataAnnotations):
                            continue
                        # Override the load_into_memory flag, because we need the contents
                        if not load_into_memory:
                            dp.metadata[fld] = Path(dp.metadata[fld]).read_bytes()
                        try:
                            dp.metadata[fld] = MetadataAnnotations.from_ls_task(
                                datapoint=dp, field=fld, ls_task=dp.metadata[fld]
                            )
                        except ValidationError:
                            bad_annotations[fld].append(dp.path)
                    else:
                        dp.metadata[fld] = MetadataAnnotations(datapoint=dp, field=fld)

        if bad_annotations:
            log_message(
                "Warning: The following datapoints had invalid annotations, "
                "any annotation-related operations will not work on these:"
            )
            err_msg = ""
            for fld, dps in bad_annotations.items():
                err_msg += f'Field "{fld}" in datapoints:\n\t'
                err_msg += "\n\t".join(dps)
            log_message(err_msg)

    def download_binary_columns(
        self,
        *columns: str,
        load_into_memory=True,
        cache_on_disk=True,
        num_proc: int = 32,
    ) -> "QueryResult":
        """
        deprecated: Use get_blob_fields instead.

        :meta private:
        """
        return self.get_blob_fields(
            *columns,
            load_into_memory=load_into_memory,
            cache_on_disk=cache_on_disk,
            num_proc=num_proc,
        )

    def annotate_with_mlflow_model(
        self,
        repo: str,
        name: str,
        host: Optional[str] = None,
        version: str = "latest",
        pre_hook: Callable[[List[str]], Any] = identity_func,
        post_hook: Callable[[Any], Any] = identity_func,
        batch_size: int = 1,
        log_to_field: str = "annotation",
    ) -> Dict[str, Any]:
        """
        Fetch an MLflow model from a specific repository and use it to annotate the datapoints in this QueryResult.
        The resulting annotations are then stored in the field specified by ``log_to_field``.

        Any MLflow model that has a ``model.predict`` endpoint is supported.
        This includes, but is not limited to the following flavors:

        * ``torch``
        * ``tensorflow``
        * ``pyfunc``
        * ``scikit-learn``

        Keep in mind that by default ``mlflow.predict()`` will receive the list of downloaded datapoint paths as input,
        so additional "massaging" of the data might be required for prediction to work.
        Use the ``pre_hook`` function to do so.

        Args:
            repo: repository to extract the model from
            name: name of the model in the repository's MLflow registry.
            host: address of the DagsHub instance with the repo to load the model from.
                 Set it if the model is hosted on a different DagsHub instance than the datasource.
            version: version of the model in the mlflow registry.
            pre_hook: function that runs before datapoints are sent to ``model.predict()``.
                The input argument is the list of paths to datapoint files in the current batch.
            post_hook: function that converts the model output to the desired format.
            batch_size: Size of the file batches that are sent to ``model.predict()``.
                Default batch size is 1, but it is still being sent as a list for consistency.
            log_to_field: Field to store the resulting annotations in.
        """
        res = self.predict_with_mlflow_model(
            repo,
            name,
            host=host,
            version=version,
            pre_hook=pre_hook,
            post_hook=post_hook,
            batch_size=batch_size,
            log_to_field=log_to_field,
        )
        self.datasource.metadata_field(log_to_field).set_annotation().apply()
        return res

    def predict_with_mlflow_model(
        self,
        repo: str,
        name: str,
        host: Optional[str] = None,
        version: Optional[str] = "latest",
        pre_hook: Callable[[List[str]], Any] = identity_func,
        post_hook: Callable[[Any], Any] = identity_func,
        batch_size: int = 1,
        log_to_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch an MLflow model from a specific repository and use it to predict on the datapoints in this QueryResult.

        Any MLflow model that has a ``model.predict`` endpoint is supported.
        This includes, but is not limited to the following flavors:

        * ``torch``
        * ``tensorflow``
        * ``pyfunc``
        * ``scikit-learn``

        Keep in mind that by default ``mlflow.predict()`` will receive the list of downloaded datapoint paths as input,
        so additional "massaging" of the data might be required for prediction to work.
        Use the ``pre_hook`` function to do so.

        Args:
            repo: repository to extract the model from
            name: name of the model in the repository's MLflow registry.
            host: address of the DagsHub instance with the repo to load the model from.
                 Set it if the model is hosted on a different DagsHub instance than the datasource.
            version: version of the model in the mlflow registry.
            pre_hook: function that runs before datapoints are sent to ``model.predict()``.
                The input argument is the list of paths to datapoint files in the current batch.
            post_hook: function that converts the model output to the desired format.
            batch_size: Size of the file batches that are sent to ``model.predict()``.
                Default batch size is 1, but it is still being sent as a list for consistency.
            log_to_field: If set, writes prediction results to this metadata field in the datasource.
        """
        if not host:
            host = self.datasource.source.repoApi.host

        model = get_mlflow_model(repo, name, host, version)
        return self.generate_predictions(lambda x: post_hook(model.predict(pre_hook(x))), batch_size, log_to_field)

    def get_annotations(self, **kwargs) -> "QueryResult":
        """
        Loads all annotation fields using :func:`get_blob_fields`.

        All keyword arguments are passed to :func:`get_blob_fields`.
        """
        if len(self.annotation_fields) == 0:
            return self

        log_message(f"Downloading annotation fields {self.annotation_fields}...")
        self.get_blob_fields(*self.annotation_fields, **kwargs)

        return self

    def download_files(
        self,
        target_dir: Optional[Union[str, PathLike]] = None,
        keep_source_prefix=True,
        redownload=False,
        path_field: Optional[str] = None,
    ) -> PathLike:
        """
        Downloads the datapoints to the ``target_dir`` directory

        Args:
            target_dir: Where to download the files. Defaults to \
                :func:`datasource's default location \
                <dagshub.data_engine.model.datasource.Datasource.default_dataset_location>`
            keep_source_prefix: If True, includes the prefix of the datasource in the download path.
            redownload: Whether to redownload a file if it exists on the filesystem already.\
                We don't do any hashsum checks, so if it's possible that the file has been updated, set to True
            path_field: Set this to the name of the field with the file's path\
                if you want to download files from a field other than the datapoint's path.

        .. note::
            For ``path_field`` the path in the field still needs to be in the same repo
            and have the same format as the path of the datapoint, including not having the prefix.
            For now, you can't download arbitrary paths/urls.

        Returns:
            Path to the directory with the downloaded files
        """

        target_path = self.datasource.default_dataset_location if target_dir is None else Path(target_dir).expanduser()
        logger.warning(f"Downloading {len(self.entries)} files to {str(target_path)}")

        if self.datasource.source.source_type == DatasourceType.BUCKET:
            send_analytics_event(
                "Client_DataEngine_downloadDatapointsFromBucket",
                repo=self.datasource.source.repoApi,
            )
        else:
            send_analytics_event(
                "Client_DataEngine_downloadDatapoints",
                repo=self.datasource.source.repoApi,
            )

        def dp_path(dp: Datapoint):
            if path_field is not None:
                path_val = dp.metadata.get(path_field)
                if path_val is None:
                    return None
            else:
                path_val = dp.path

            if keep_source_prefix:
                return target_path / self.datasource.source.source_prefix / path_val
            else:
                return target_path / path_val

        def dp_url(dp: Datapoint):
            if path_field is not None:
                path_val = dp.metadata.get(path_field)
                if path_val is None:
                    return None
                return self.datasource.source.raw_path(path_val)
            else:
                return dp.download_url

        download_args = [(dp_url(dp), dp_path(dp)) for dp in self.entries if dp_path(dp) is not None]

        download_files(download_args, skip_if_exists=not redownload)
        return target_path

    def _get_all_annotations(self, annotation_field: str) -> List[IRImageAnnotationBase]:
        annotations = []
        for dp in self.entries:
            if annotation_field in dp.metadata:
                annotations.extend(dp.metadata[annotation_field].annotations)
        return annotations

    def export_as_yolo(
        self,
        download_dir: Optional[Union[str, Path]] = None,
        annotation_field: Optional[str] = None,
        annotation_type: Optional[Literal["bbox", "segmentation", "pose"]] = None,
    ) -> Path:
        """
        Downloads the files and annotations in a way that can be used to train with YOLO immediately.

        Args:
            download_dir: Where to download the files. Defaults to ``./dagshub_export``
            annotation_field: Field with the annotations. If None, uses the first alphabetical annotation field.
            annotation_type: Type of YOLO annotations to export.
                Possible values: "bbox", "segmentation", "pose".
                If None, returns based on the most common annotation type.

        Returns:
            The path to the YAML file with the metadata. Pass this path to ``YOLO.train()`` to train a model.
        """
        if annotation_field is None:
            annotation_field = sorted([f.name for f in self.fields if f.is_annotation()])
            if len(annotation_field) == 0:
                raise ValueError("No annotation fields found in the datasource")
            annotation_field = annotation_field[0]
            log_message(f"Using annotations from field {annotation_field}")

        if download_dir is None:
            download_dir = Path("dagshub_export")
        download_dir = Path(download_dir) / "data"

        annotations = self._get_all_annotations(annotation_field)

        categories = Categories()
        ann_types: dict[type, int] = {}

        for ann in annotations:
            if annotation_type is None:
                ann_type = type(ann)
                ann_types[ann_type] = ann_types.get(ann_type, 0) + 1
            for cat in ann.categories.keys():
                categories.get_or_create(cat)

        if annotation_type is None:
            annotation_types = [type(ann) for ann in annotations]
            counter = Counter(annotation_types)
            most_common = counter.most_common(1)
            annotation_type = ir_mapping[most_common[0][0]]
            log_message(f"Importing annotations of type {annotation_type} (most common type of annotation).")

        image_download_path = Path(download_dir)

        # Add the source prefix to all annotations
        for ann in annotations:
            ann.filename = os.path.join(self.datasource.source.source_prefix, ann.filename)

        # If there's no folder "images" in the datasource, prepend it to the path
        if not any("images/" in ann.filename for ann in annotations):
            image_download_path = Path(download_dir) / "images"

        context = YoloContext(annotation_type=annotation_type, categories=categories)
        context.path = image_download_path

        log_message("Downloading image files...")
        self.download_files(image_download_path)
        log_message("Exporting annotations...")
        yaml_path = dagshub_annotation_converter.converters.yolo.export_to_fs(context, annotations)
        log_message(f"Done! Saved YOLO Dataset, YAML file is at {yaml_path.absolute()}")
        return yaml_path

    def to_voxel51_dataset(self, **kwargs) -> "fo.Dataset":
        """
        Creates a voxel51 dataset that can be used with\
        `fo.launch_app()
        <https://docs.voxel51.com/api/fiftyone.core.session.html?highlight=launch_app#fiftyone.core.session.launch_app>`_
        to visualize it.

        Keyword Args:
            name (str): Name of the dataset. Default is the name of the datasource.
            force_download (bool): Download the dataset even if the size of the files is bigger than 100MB.\
                Default is False
            files_location (str|PathLike): path to the location where to download the local files.
                Default is :func:`datasource's default location \
                <dagshub.data_engine.model.datasource.Datasource.default_dataset_location>`
            redownload (bool): Redownload files, replacing the ones that might exist on the filesystem.\
                Default is False.
            voxel_annotations (List[str]) : List of fields from which to load voxel annotations serialized with
                                        `to_json()`. This will override the labelstudio annotations

        :rtype: `fo.Dataset <https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset>`_
        """
        if len(self.entries) == 0:
            raise VisualizeError("No datapoints to visualize")
        logger.info("Migrating dataset to voxel51")
        name = kwargs.get("name", self.datasource.source.name)
        force_download = kwargs.get("force_download", False)
        # TODO: don't override samples in existing one
        if fo.dataset_exists(name):
            ds: fo.Dataset = fo.load_dataset(name)
        else:
            ds: fo.Dataset = fo.Dataset(name)
        # ds.persistent = True

        dataset_location = Path(kwargs.get("files_location", self.datasource.default_dataset_location))
        dataset_location.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading files...")

        if "voxel_annotations" in kwargs:
            annotation_fields = kwargs["voxel_annotations"]
            label_func = add_voxel_annotations
        else:
            annotation_fields = [f for f in self.fields if f.is_annotation()]
            label_func = add_ls_annotations

        # Load the annotation fields
        self.download_binary_columns(*annotation_fields)

        if not force_download:
            self._check_downloaded_dataset_size()

        redownload = kwargs.get("redownload", False)
        self.download_files(self.datasource.default_dataset_location, redownload=redownload)

        progress = get_rich_progress(rich.progress.MofNCompleteColumn())
        task = progress.add_task("Generating voxel samples...", total=len(self.entries))

        samples: List["fo.Sample"] = []

        with progress:
            for datapoint in self.entries:
                filepath = self.datasource.default_dataset_location / datapoint.path_in_repo
                sample = fo.Sample(filepath=filepath)
                sample["dagshub_download_url"] = datapoint.download_url
                sample["datapoint_id"] = datapoint.datapoint_id
                sample["datapoint_path"] = datapoint.path
                label_func(sample, datapoint, *annotation_fields)
                for k, v in datapoint.metadata.items():
                    # TODO: more filtering here, not all fields should be showing up in voxel
                    if k in annotation_fields:
                        continue
                    if type(v) is not bytes:
                        sample[k] = v
                samples.append(sample)
                progress.update(task, advance=1, refresh=True)

        ds.merge_samples(samples)
        return ds

    @staticmethod
    def _get_predict_dict(predictions, remote_path, log_to_field):
        res = {log_to_field: json.dumps(predictions[remote_path][0]).encode("utf-8")}
        if len(predictions[remote_path]) == 2:
            res[f"{log_to_field}_score"] = predictions[remote_path][1]

        return res

    def _check_downloaded_dataset_size(self):
        download_size_prompt_threshold = 100 * (2**20)  # 100 Megabytes
        dp_size = self._calculate_datapoint_size()
        if dp_size is not None and dp_size > download_size_prompt_threshold:
            prompt = f"You're about to download {sizeof_fmt(dp_size)} of images locally."
            should_download = prompt_user(prompt)
            if not should_download:
                msg = "Downloading voxel dataset cancelled"
                logger.warning(msg)
                raise RuntimeError(msg)

    def _calculate_datapoint_size(self) -> Optional[int]:
        sum_size = 0
        has_sum_field = False
        all_have_sum_field = True
        size_field = "size"
        for dp in self.entries:
            if size_field in dp.metadata:
                has_sum_field = True
                sum_size += dp.metadata[size_field]
            else:
                all_have_sum_field = False
        if not has_sum_field:
            logger.warning("None of the datapoints had a size field, can't calculate size of the downloading dataset")
            return None
        if not all_have_sum_field:
            logger.warning("Not every datapoint has a size field, size calculations might be wrong")
        return sum_size

    def visualize(self, visualizer: Literal["dagshub", "fiftyone"] = "dagshub", **kwargs) -> Union[str, "fo.Session"]:
        """
        Visualize this QueryResult either on DagsHub or with Voxel51.

        If ``visualizer`` is ``dagshub``, a webpage is opened on DagsHub with the query applied.

        If ``visualizer`` is ``fiftyone``, this function calls :func:`to_voxel51_dataset`,
        passing to it the keyword arguments, and launches a fiftyone session showing the dataset.

        Additionally, this function adds a DagsHub plugin into Voxel51 that you can use for additional interactions
        with the datasource from within the voxel environment.

        Returns the session object that you can ``wait()`` if you are using it\
        outside a notebook and need to not close the script immediately::

            session = ds.all().visualize()
            session.wait(-1)
        """
        if visualizer == "dagshub":
            return self.datasource.visualize(visualizer, **kwargs)
        if visualizer == "fiftyone":
            set_voxel_envvars()

            send_analytics_event("Client_DataEngine_VizualizeResults", repo=self.datasource.source.repoApi)

            ds = self.to_voxel51_dataset(**kwargs)

            sess = fo.launch_app(ds)
            # Launch the server for plugin interaction
            plugin_server_module.run_plugin_server(sess, self.datasource, self.datasource.source.revision)

            return sess

    def generate_predictions(
        self,
        predict_fn: CustomPredictor,
        batch_size: int = 1,
        log_to_field: Optional[str] = None,
    ) -> Dict[str, Tuple[str, Optional[float]]]:
        """
        Sends all the datapoints returned in this QueryResult as prediction targets for
        a generic object.

        Args:
            predict_fn: function that handles batched input and returns predictions with an optional prediction score.
            batch_size: (optional, default: 1) number of datapoints to run inference on simultaneously
            log_to_field: (optional, default: 'prediction') write prediction results to metadata logged in data engine.
            If None, just returns predictions.
            (in addition to logging to a field, iff that parameter is set)
        """
        dset = DagsHubDataset(self, tensorizers=[lambda x: x])

        predictions = {}
        progress = get_rich_progress(rich.progress.MofNCompleteColumn())
        task = progress.add_task("Running inference...", total=len(dset))
        with progress:
            for idx, local_paths in enumerate(
                _Batcher(dset, batch_size) if batch_size != 1 else dset
            ):  # encapsulates dataset with batcher if necessary and iterates over it
                for prediction, remote_path in zip(
                    predict_fn(local_paths),
                    [result.path for result in self[idx * batch_size : (idx + 1) * batch_size]],
                ):
                    predictions[remote_path] = prediction
                progress.update(task, advance=batch_size, refresh=True)

        if log_to_field:
            with self.datasource.metadata_context() as ctx:
                for remote_path in predictions:
                    ctx.update_metadata(remote_path, self._get_predict_dict(predictions, remote_path, log_to_field))
        return predictions

    def generate_annotations(self, predict_fn: CustomPredictor, batch_size: int = 1, log_to_field: str = "annotation"):
        """
        Sends all the datapoints returned in this QueryResult as prediction targets for
        a generic object.

        Args:
            predict_fn: function that handles batched input and returns predictions with an optional prediction score.
            batch_size: (optional, default: 1) number of datapoints to run inference on simultaneously.
            log_to_field: (optional, default: 'prediction') write prediction results to metadata logged in data engine.
        """
        self.generate_predictions(
            predict_fn,
            batch_size=batch_size,
            log_to_field=log_to_field,
        )
        self.datasource.metadata_field(log_to_field).set_annotation().apply()

    def annotate(
        self,
        open_project: bool = True,
        ignore_warning: bool = True,
        fields_to_embed: Optional[List[str]] = None,
        fields_to_exclude: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Sends all the datapoints returned in this QueryResult to be annotated in Label Studio on DagsHub.
        Alternatively, uses MLFlow to automatically label datapoints.

        Args:
            open_project: Automatically open the Label Studio project in the browser
            ignore_warning: Suppress the prompt-warning if you try to annotate too many datapoints at once.
            fields_to_embed: list of meta-data columns that will show up in Label Studio UI.
             if not specified all will be displayed.
            fields_to_exclude: list of meta-data columns that will not show up in Label Studio UI
        Returns:
            The URL of the created Label Studio workspace
        """
        send_analytics_event("Client_DataEngine_SentToAnnotation", repo=self.datasource.source.repoApi)

        return self.datasource.send_datapoints_to_annotation(
            self.entries,
            open_project=open_project,
            ignore_warning=ignore_warning,
            fields_to_exclude=fields_to_exclude,
            fields_to_embed=fields_to_embed,
        )

    @property
    def document_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.is_document()]

    @property
    def annotation_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.is_annotation()]

    def _load_autoload_fields(self, documents=True, annotations=True):
        """
        Loads fields that are supposed to be load automatically upon querying.
        This includes:
            - All document fields
            - All annotation fields
        """
        if documents:
            if len(self.document_fields) > 0:
                self.get_blob_fields(*self.document_fields, load_into_memory=True)

        if annotations:
            self.get_annotations()

    def log_to_mlflow(self, run: Optional["mlflow.entities.Run"] = None) -> "mlflow.entities.Run":
        """
        Logs the query result information to MLflow as an artifact.
        The artifact will be saved at the root of the run with the name in the format of
        ``log_{datasource_name}_{query_time}_{random_chunk}.dagshub.dataset.json``.

        You can later load the dataset back from MLflow using :func:`dagshub.data_engine.datasources.get_from_mlflow`.

        Args:
            run: MLflow run to save to. If ``None``, uses the active MLflow run or creates a new run.

        Returns:
            Run to which the artifact was logged.
        """
        assert self.query_data_time is not None
        artifact_name = self.datasource._get_mlflow_artifact_name("log", self.query_data_time)
        return self.datasource._log_to_mlflow(artifact_name, run, self.query_data_time)


# to support depedency-free dataloading, `_Batcher` is a barebones dataloader that sets up batched inference
class _Batcher:
    def __init__(self, dset, batch_size):
        self.dset = dset
        self.batch_size = batch_size

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __next__(self):
        self.curr_idx += self.batch_size
        return [self.dset[idx] for idx in range(self.curr_idx - self.batch_size, self.curr_idx)]
