from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any

from dagshub.common.download import download_files
from dagshub.common.helpers import http_request


@dataclass
class Datapoint:
    datapoint_id: int
    path: str
    metadata: Dict[str, Any]
    datasource: "Datasource"

    def __getitem__(self, item):
        return self.metadata[item]

    @property
    def download_url(self):
        return self.datasource.source.raw_path(self)

    @property
    def path_in_repo(self):
        return self.datasource.source.file_path(self)

    @staticmethod
    def from_gql_edge(edge: Dict, datasource: "Datasource") -> "Datapoint":
        res = Datapoint(
            datapoint_id=int(edge["node"]["id"]),
            path=edge["node"]["path"],
            metadata={},
            datasource=datasource,
        )
        for meta_dict in edge["node"]["metadata"]:
            res.metadata[meta_dict["key"]] = meta_dict["value"]
        return res

    def to_dict(self, metadata_keys: List[str]) -> Dict[str, Any]:
        res_dict = {
            "name": self.path,
            "datapoint_id": self.datapoint_id,
            "dagshub_download_url": self.download_url,
        }
        res_dict.update({key: self.metadata.get(key) for key in metadata_keys})
        return res_dict

    def get_blob(self, column: str, cache_on_disk=True, store_value=False) -> bytes:
        """
        Returns the blob stored in the binary column

        Args:
            column: where to get the blob from
            cache_on_disk: whether to store the downloaded blob on the disk or not
            store_value: whether to store the blob in the field after acquiring it
        """
        current_value = self.metadata[column]

        if type(current_value) is bytes:
            # Bytes - it's already there!
            return current_value
        if isinstance(current_value, Path):
            # Path - assume the path exists and is already downloaded,
            #   because it's unlikely that the user has set it themselves
            with current_value.open("rb") as f:
                content = f.read()
            if store_value:
                self.metadata[column] = content
            return content

        elif type(current_value) is str:
            # String - This is probably the hash of the blob, get that from dagshub
            blob_url = self.blob_url(current_value)
            blob_location = self.blob_cache_location / current_value

            # Make sure that the cache location exists
            if cache_on_disk:
                self.blob_cache_location.mkdir(parents=True, exist_ok=True)

            content = _get_blob(blob_url, blob_location, self.datasource.source.repoApi.auth, cache_on_disk, True)
            if type(content) is str:
                raise RuntimeError(f"Error while downloading blob: {content}")

            if store_value:
                self.metadata[column] = content
            elif cache_on_disk:
                self.metadata[column] = blob_location

            return content
        else:
            raise ValueError(f"Can't extract blob metadata from value {current_value} of type {type(current_value)}")

    def download_file(self, target: Optional[Union[PathLike, str]] = None, keep_source_prefix=True,
                      redownload=False) -> PathLike:
        """
        Downloads the datapoint to the target_dir directory
        Args:
            target_dir: Where to download the file (either a directory, or the full path).
                If not specified, then downloads to ~/dagshub/datasets/<user>/<repo>/<ds_id>
            keep_source_prefix: If True, includes the prefix of the datasource in the download path
                Useful for cases where the download path is the root of the repository
            redownload: Whether to redownload a file if it exists on the filesystem already
                NOTE: We don't do any hashsum checks, so if it's possible that the file has been updated, turn it on
        Returns:
            Path to the downloaded file
        """

        target_path = self.datasource.default_dataset_location if target is None else Path(target).expanduser()

        # Check if the specified path looks like a file
        # by checking if there's an extension, or it's an already existing file
        n = target_path.name
        target_is_already_file = (target_path.exists() and target_path.is_file()) or (
            "." in n and not n.startswith("."))

        if not target_is_already_file:
            if keep_source_prefix:
                target_path = target_path / self.path_in_repo
            else:
                target_path = target_path / self.path

        download_files([(self.download_url, target_path)], skip_if_exists=not redownload)
        return target_path

    @property
    def blob_cache_location(self):
        return self.datasource.default_dataset_location / ".metadata_blobs"

    def blob_url(self, sha):
        return self.datasource.source.blob_path(sha)

    def _extract_blob_url_and_path(self, col: str) -> Tuple[Optional[str], Optional[PathLike]]:
        sha = self.metadata.get(col)
        if sha is None or type(sha) is not str:
            return None, None
        return self.blob_url(sha), self.blob_cache_location / sha


def _get_blob(url: Optional[str], cache_path: Optional[Path], auth, cache_on_disk, return_blob) -> Optional[
    Union[Path, str, bytes]]:
    """
    Args:
        url: url to download the blob from
        cache_path: where the cache for the blob is (laods from it if exists, stores there if it doesn't)
        auth: auth to use for getting the blob
        cache_on_disk: whether to store the downloaded blob on disk. If False we also turn off the cache checking
        return_blob: if True returns the blob of the downloaded data, if False returns the path to the file with it
    """
    if url is None:
        return None
    assert cache_path is not None

    if cache_on_disk and cache_path.exists():
        if return_blob:
            with cache_path.open("rb") as f:
                return f.read()
        else:
            return cache_path

    try:
        # TODO: add retries here
        resp = http_request("GET", url, auth=auth)
        if resp.status_code >= 400:
            return f"Error while downloading binary blob (Status code {resp.status_code}): {resp.content.decode()}"
        content = resp.content
    except Exception as e:
        return f"Error while downloading binary blob: {e}"

    if cache_on_disk:
        with cache_path.open("wb") as f:
            f.write(content)

    if return_blob:
        return content
    else:
        return cache_path
