import logging
from os import PathLike
from pathlib import Path, PurePosixPath
import rich.progress
import json

from dagshub.common.api.responses import (
    RepoAPIResponse,
    BranchAPIResponse,
    CommitAPIResponse,
    StorageAPIEntry,
    ContentAPIEntry,
    StorageContentAPIResult,
)
from dagshub.data_engine.model.errors import LSInitializingError
from dagshub.common.download import download_files
from dagshub.common.rich_util import get_rich_progress
from dagshub.common.util import multi_urljoin
from functools import partial

from pydantic import BaseModel

from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from functools import cached_property

from typing import Optional, Tuple, Any, List, Union, Dict

import dacite

import dagshub.auth
from dagshub.common import config

from dagshub.common.helpers import http_request, log_message

logger = logging.getLogger("dagshub")


class WrongRepoFormatError(Exception):
    pass


class RepoNotFoundError(Exception):
    pass


class BranchNotFoundError(Exception):
    pass


class CommitNotFoundError(Exception):
    pass


class PathNotFoundError(Exception):
    pass


class LabelStudioProject(BaseModel):
    project_name: str
    project_id: str


class RepoAPI:
    def __init__(self, repo: str, host: Optional[str] = None, auth: Optional[Any] = None):
        """
        Class for interacting with the API of a repository

        Args:
            repo: repo in the format of ``user/repo``
            host (optional): url of the DagsHub instance the repo is on
            auth (optional): authentication to use to connect
        """
        self.owner, self.repo_name = self.parse_repo(repo)
        self.host = host if host is not None else config.host

        if auth is None:
            self.auth = dagshub.auth.get_authenticator(host=host)
        else:
            self.auth = auth

    @retry(retry=retry_if_exception_type(LSInitializingError), wait=wait_fixed(3), stop=stop_after_attempt(5))
    def _tenacious_ls_request(self, *args, **kwargs):
        res = http_request(*args, **kwargs)
        if res.text.startswith("<!DOCTYPE html>"):
            raise LSInitializingError()
        elif res.status_code // 100 != 2:
            raise RuntimeError(f"Process failed! Server Response: {res.text}")
        return res

    def _http_request(self, method, url, **kwargs):
        if "auth" not in kwargs:
            kwargs["auth"] = self.auth
        return http_request(method, url, **kwargs)

    def get_repo_info(self) -> RepoAPIResponse:
        """
        Get information about the repository
        """
        res = self._http_request("GET", self.repo_api_url)

        if res.status_code == 404:
            raise RepoNotFoundError(f"Repo {self.repo_url} not found")
        elif res.status_code >= 400:
            error_msg = f"Got status code {res.status_code} when getting repository info."
            logger.error(error_msg)
            logger.debug(res.content)
            raise RuntimeError(error_msg)
        return dacite.from_dict(RepoAPIResponse, res.json())

    def get_branch_info(self, branch: str) -> BranchAPIResponse:
        """
        Get information about specified branch

        Args:
            branch: Name of the branch to get the info
        """
        res = self._http_request("GET", self.branch_url(branch))

        if res.status_code == 404:
            raise BranchNotFoundError(f"Branch {branch} not found in repo {self.repo_url}")
        elif res.status_code >= 400:
            error_msg = f"Got status code {res.status_code} when getting branch."
            logger.error(error_msg)
            logger.debug(res.content)
            raise RuntimeError(error_msg)

        return dacite.from_dict(BranchAPIResponse, res.json())

    def get_commit_info(self, sha: str) -> CommitAPIResponse:
        """
        Get information about a specific commit

        Args:
            sha: SHA of the commit.
        """
        res = self._http_request("GET", self.commit_url(sha))

        if res.status_code == 404:
            raise BranchNotFoundError(f"Commit {sha} not found in repo {self.repo_url}")
        elif res.status_code >= 400:
            error_msg = f"Got status code {res.status_code} when getting commit."
            logger.error(error_msg)
            logger.debug(res.content)
            raise RuntimeError(error_msg)

        return dacite.from_dict(CommitAPIResponse, res.json()["commit"])

    def get_connected_storages(self) -> List[StorageAPIEntry]:
        """
        Get storages that are connected to the repository
        """
        res = self._http_request("GET", self.storage_api_url())

        if res.status_code == 404:
            raise RepoNotFoundError(f"Repo {self.repo_url} not found")
        elif res.status_code >= 400:
            error_msg = f"Got status code {res.status_code} when getting repository info."
            logger.error(error_msg)
            logger.debug(res.content)
            raise RuntimeError(error_msg)

        return [dacite.from_dict(StorageAPIEntry, storage_entry) for storage_entry in res.json()]

    def list_annotation_projects(self) -> Dict[str, LabelStudioProject]:
        """
        Get annotation projects that are associated with the repository
        """
        res = self._tenacious_ls_request(
            "GET",
            multi_urljoin(self.label_studio_api_url(), "projects"),
            headers={"Authorization": f"Bearer {dagshub.auth.get_token()}"},
        )
        return {
            project["title"]: LabelStudioProject(project_name=project["title"], project_id=str(project["id"]))
            for project in res.json()["results"]
        }

    def add_annotation_project(self, project_name: str, config: Optional[str] = None) -> None:
        """
        Add an annotation project to the repository
        """
        if project_name in self.list_annotation_projects():
            raise ValueError(f"{project_name} already exists!")

        self._tenacious_ls_request(
            "POST",
            multi_urljoin(self.label_studio_api_url(), "projects"),
            headers={"Authorization": f"Bearer {dagshub.auth.get_token()}", "Content-Type": "application/json"},
            data=json.dumps(
                {"title": project_name, "label_config": config} if config is not None else {"title": project_name}
            ),
        )

    def update_label_studio_project_config(self, project_name: str, config: str) -> None:
        """
        Update the labelling config of an annotation project
        """
        projects = self.list_annotation_projects()
        if project_name not in projects:
            raise ValueError(f"{project_name} doesn't exist!")

        self._tenacious_ls_request(
            "PATCH",
            multi_urljoin(self.label_studio_api_url(), "projects", projects[project_name].project_id),
            headers={"Authorization": f"Bearer {dagshub.auth.get_token()}", "Content-Type": "application/json"},
            data=json.dumps({"title": project_name, "label_config": config}),
        )

    def add_autolabelling_endpoint(self, project_name: str, endpoint: str) -> None:
        """
        Add an endpoint serving MLBackend for auto-labelling
        """
        projects = self.list_annotation_projects()
        if project_name not in projects:
            raise ValueError(f"{project_name} not in projects. Available project names: {list(projects.keys())}")

        self._tenacious_ls_request(
            "POST",
            multi_urljoin(self.label_studio_api_url(), "ml"),
            headers={"Authorization": f"Bearer {dagshub.auth.get_token()}", "Content-Type": "application/json"},
            data=json.dumps({"url": endpoint, "project": projects[project_name].project_id}),
        )

    def list_path(self, path: str, revision: Optional[str] = None, include_size: bool = False) -> List[ContentAPIEntry]:
        """
        List contents of a repository directory

        Args:
            path: Path of the directory
            revision: Branch or commit SHA. Default is tip of the default branch.
            include_size: Whether to include sizes for files. Calculating sizes might take more time.
        """
        params = {"include_size": include_size}
        res = self._http_request("GET", self.content_api_url(path, revision), params=params)

        if res.status_code == 404:
            raise PathNotFoundError(f"Path {path} not found")
        elif res.status_code >= 400:
            error_msg = f"Got status code {res.status_code} when listing path {path}"
            logger.error(error_msg)
            logger.debug(res.content)
            raise RuntimeError(error_msg)

        content = res.json()
        if type(content) is dict:
            content = [content]
        return [dacite.from_dict(ContentAPIEntry, entry) for entry in content]

    def list_storage_path(self, path: str, include_size: bool = False) -> List[ContentAPIEntry]:
        """
        List contents of a folder in a connected storage bucket

        Args:
            path: Path of the storage directory in the format of ``<scheme>/<bucket-name>/<path>``

        Path example: ``s3/my-bucket/prefix/path/to/file``
        """
        params = {"include_size": include_size, "paging": True}

        url = self.storage_content_api_url(path)
        has_next_page = True

        def _get():
            res = self._http_request("GET", url, params=params)

            if res.status_code == 404:
                raise PathNotFoundError(f"Path {path} not found")
            elif res.status_code >= 400:
                error_msg = f"Got status code {res.status_code} when listing path {path}"
                logger.error(error_msg)
                logger.debug(res.content)
                raise RuntimeError(error_msg)

            content = res.json()
            if "entries" not in content:
                content = {
                    "entries": [
                        content,
                    ],
                    "next_token": None,
                }
            return dacite.from_dict(StorageContentAPIResult, content)

        entries = []

        while has_next_page:
            has_next_page = False
            resp = _get()
            entries += resp.entries
            if resp.next_token is not None:
                has_next_page = True
                params["from_token"] = resp.next_token

        return entries

    def get_file(self, path: str, revision: Optional[str] = None) -> bytes:
        """
        Download file from repo.

        Args:
            path: Path of the file in the repo.
            revision: Git branch or revision from which to download the file.

        Returns:
            bytes: The content of the file.
        """
        res = self._http_request("GET", self.raw_api_url(path, revision))
        if res.status_code == 404:
            raise PathNotFoundError(f"Path {path} not found")
        elif res.status_code >= 400:
            error_msg = f"Got status code {res.status_code} when getting file {path}"
            logger.error(error_msg)
            logger.debug(res.content)
            raise RuntimeError(error_msg)
        return res.content

    def get_storage_file(self, path: str) -> bytes:
        """
        Download file from a connected storage bucket.

        Args:
            path: Path in the bucket in the format of ``<scheme>/<bucket-name>/<path>``.

        Path example: ``s3/my-bucket/prefix/path/to/file``

        Returns:
            bytes: The content of the file.
        """
        res = self._http_request("GET", self.storage_raw_api_url(path))
        if res.status_code == 404:
            raise PathNotFoundError(f"Path {path} not found")
        elif res.status_code >= 400:
            error_msg = f"Got status code {res.status_code} when getting file {path}"
            logger.error(error_msg)
            logger.debug(res.content)
            raise RuntimeError(error_msg)
        return res.content

    def _get_files_in_path(
        self, path, revision=None, recursive=False, traverse_storages=False
    ) -> List[ContentAPIEntry]:
        """
        Walks through the path of the repo, returning non-dir entries
        """

        dir_queue = []
        files = []

        list_fn_folder = partial(self.list_path, revision=revision)
        list_fn_storage = self.list_storage_path

        def push_folder(folder_path):
            dir_queue.append((folder_path, list_fn_folder))

        def push_storage(storage_path):
            dir_queue.append((storage_path, list_fn_storage))

        # Initialize the queue
        path, is_storage_path = self._sanitize_storage_path(path)
        if is_storage_path:
            push_storage(path)
        else:
            push_folder(path)

        progress = get_rich_progress(rich.progress.MofNCompleteColumn())
        task = progress.add_task("Traversing directories...", total=None)

        def step(step_path, list_fn):
            """
            step_path: path in the repo to list
            list_fn: which function to use to list it (can be list_path or list_storage_path)
            """
            res = list_fn(step_path)
            for entry in res:
                if entry.type == "file":
                    files.append(entry)
                elif recursive:
                    if entry.versioning == "bucket":
                        if traverse_storages:
                            push_storage(entry.path)
                    else:
                        push_folder(entry.path)
            progress.update(task, advance=1)

        with progress:
            while len(dir_queue):
                query_path, list_fn = dir_queue.pop(0)
                step(query_path, list_fn)

        return files

    def download(
        self,
        remote_path: Union[str, PathLike],
        local_path: Union[str, PathLike] = ".",
        revision: Optional[str] = None,
        recursive=True,
        keep_source_prefix=False,
        redownload=False,
        download_storages=False,
    ):
        """
        Downloads the contents of the repository at "remote_path" to the "local_path"

        Args:
            remote_path: Path in the repository of the folder or file to download.
            local_path: Where to download the files. Defaults to current working directory.
            revision: Repo revision or branch, if not specified - uses default repo branch.
                Ignored for downloading from buckets.
            recursive: Whether to download files recursively.
            keep_source_prefix: | Whether to keep the path of the folder in the download path or not.
                | Example: Given remote_path ``src/data`` and file ``test/file.txt``
                | if ``True``: will download to ``<local_path>/src/data/test/file.txt``
                | if ``False``: will download to ``<local_path>/test/file.txt``
            redownload: Whether to redownload files that already exist on the local filesystem.
                The downloader doesn't do any hash comparisons and only checks
                if a file already exists in the local filesystem or not.
            download_storages: If downloading the whole repo, by default we're not downloading the integrated storages
                Toggle this to ``True`` to change this behavior
        """
        traverse_storages = True
        if str(remote_path) == "/" and not download_storages:
            log_message(
                "Skipping downloading from connected storages. "
                "Set the `download_storages` flag if you want "
                "to download the whole content of the connected storages."
            )
            traverse_storages = False

        files = self._get_files_in_path(remote_path, revision, recursive, traverse_storages=traverse_storages)
        file_tuples = []
        if local_path is None:
            local_path = "."
        local_path = Path(local_path)
        # Strip the slashes from the beginning so the relative_to logic works
        remote_path = str(remote_path).lstrip("/")
        if not remote_path:
            remote_path = "/"
        # For storage paths get rid of the colon in the beginning of the schema, the download urls won't have it either
        remote_path, _ = self._sanitize_storage_path(remote_path)
        # Edge case - if the user requested a single file - different output path semantics
        if len(files) == 1 and files[0].path == remote_path:
            f = files[0]
            remote_path = PurePosixPath(f.path)
            # If local_path was specified, assume that the local_path is the exact name of the file
            if local_path != Path("."):
                # Saving to existing dir - append the name of remote file to the end a-la cp
                if local_path.is_dir():
                    remote_path = remote_path if keep_source_prefix else remote_path.name
                    file_path = local_path / remote_path
                else:
                    file_path = local_path
            else:
                file_path = remote_path if keep_source_prefix else remote_path.name
            file_tuples.append((f.download_url, file_path))
        else:
            for f in files:
                file_path_in_remote = PurePosixPath(f.path)
                remote_path_obj = PurePosixPath(remote_path)
                if not keep_source_prefix and remote_path != "/":
                    file_path = file_path_in_remote.relative_to(remote_path_obj)
                else:
                    file_path = file_path_in_remote
                file_path = local_path / file_path
                file_tuples.append((f.download_url, file_path))
        download_files(file_tuples, skip_if_exists=not redownload)
        log_message(f"Downloaded {len(files)} file(s) to {local_path.resolve()}")

    @staticmethod
    def _sanitize_storage_path(path: Union[str, PathLike]) -> Tuple[str, bool]:
        """
        Sanitizes storage paths for use in the traversal/download functions.
        When user asks for ``s3:/prefix/file``, we need to request ``storage/s3/prefix/file``.
        This function checks that the user has asked for a storage path (by the schema in the beginning)
        and returns a path that can be used for these types of requests.
        If the path is not a storage path, path is returned as is (converted to string).
        """
        path = str(path)
        if path.lstrip("/").split("/")[0] in {"s3:", "gs:", "azure:"}:
            path = str(path).lstrip("/").replace(":", "", 1)
            return path, True
        return path, False

    @cached_property
    def default_branch(self) -> str:
        """
        Name of the repository's default branch
        """
        return self.get_repo_info().default_branch

    @cached_property
    def id(self) -> int:
        return self.get_repo_info().id

    @cached_property
    def is_mirror(self) -> bool:
        return self.get_repo_info().mirror

    @property
    def full_name(self) -> str:
        """
        Full name of the repo in ``<owner>/<reponame>`` format
        """
        return f"{self.owner}/{self.repo_name}"

    def last_commit(self, branch: Optional[str] = None) -> CommitAPIResponse:
        """
        Returns info about the last commit of a branch.

        Args:
            branch: Branch to get the last commit of. Defaults to :func:`default_branch`.
        """
        if branch is None:
            branch = self.default_branch
        return self.get_branch_info(branch).commit

    def last_commit_sha(self, branch: Optional[str] = None) -> str:
        """
        Returns the SHA hash of the last commit of a branch.

        Args:
            branch: Branch to get the last commit of. Defaults to :func:`default_branch`.
        """
        return self.last_commit(branch).id

    @cached_property
    def repo_api_url(self) -> str:
        """
        Base URL for making all API request for the repos.
        Format: https://dagshub.com/api/v1/repos/user/repo

        :meta private:
        """
        return multi_urljoin(
            self.host,
            "api/v1/repos",
            self.owner,
            self.repo_name,
        )

    @cached_property
    def repo_url(self) -> str:
        """
        URL of the repo on DagsHub

        Format: ``https://dagshub.com/<user>/<repo>``
        """
        return multi_urljoin(self.host, self.owner, self.repo_name)

    def branch_url(self, branch) -> str:
        """
        URL of a branch on the repo
        Format: https://dasghub.com/api/v1/repos/user/repo/branches/branch

        :meta private:
        """
        return multi_urljoin(self.repo_api_url, "branches", branch)

    @cached_property
    def data_engine_url(self) -> str:
        """
        URL of data engine

        :meta private:
        """
        return multi_urljoin(self.repo_api_url, "data-engine")

    @cached_property
    def annotations_url(self) -> str:
        return multi_urljoin(self.repo_api_url, "annotations")

    def commit_url(self, sha) -> str:
        """
        URL of a commit in the repo
        Format: https://dagshub.com/api/v1/repos/user/repo/commits/sha

        :meta private:
        """
        return multi_urljoin(self.repo_api_url, "commits", sha)

    def content_api_url(self, path: str, revision: Optional[str] = None) -> str:
        """
        URL for Content API access
        Format: https://dagshub.com/api/v1/repos/user/repo/content/revision/path

        :meta private:
        """
        if revision is None:
            revision = self.default_branch
        return multi_urljoin(self.repo_api_url, "content", revision, path)

    def raw_api_url(self, path: str, revision: Optional[str] = None) -> str:
        """
        URL for Raw Content API access
        Format: https://dagshub.com/api/v1/repos/user/repo/raw/revision/path

        :meta private:
        """
        if revision is None:
            revision = self.default_branch
        return multi_urljoin(self.repo_api_url, "raw", revision, path)

    def storage_content_api_url(self, path: str) -> str:
        """
        URL for Storage Content API access
        path example: s3/bucket-name/path/in/bucket
        Format: https://dagshub.com/api/v1/repos/user/repo/storage/content/path

        :meta private:
        """
        return multi_urljoin(self.repo_api_url, "storage/content", path)

    def storage_raw_api_url(self, path: str) -> str:
        """
        URL for Storage Raw Content API access
        path example: s3/bucket-name/path/in/bucket
        Format: https://dagshub.com/api/v1/repos/user/repo/storage/raw/path

        :meta private:
        """
        return multi_urljoin(self.repo_api_url, "storage/raw", path)

    def storage_api_url(self) -> str:
        """
        URL for getting connected storages
        Format: https://dagshub.com/api/v1/repos/user/repo/storage

        :meta private:
        """
        return multi_urljoin(self.repo_api_url, "storage")

    def label_studio_api_url(self) -> str:
        """
        URL for getting label studio api
        Format: https://dagshub.com/<repo-owner>/<repo-name>/annotations/de/api/

        :meta private:
        """
        return multi_urljoin(self.host, self.full_name, "annotations/de/api/")

    def repo_bucket_api_url(self) -> str:
        """
        Endpoint URL for getting access to the S3-compatible repo bucket

        Format: https://dagshub.com/api/v1/repo-buckets/s3/user

        The bucket name is usually the name of the repo

        :meta private:
        """
        return multi_urljoin(self.host, "api/v1/repo-buckets/s3", self.owner)

    @staticmethod
    def parse_repo(repo: str) -> Tuple[str, str]:
        repo = repo.strip("/")
        parts = repo.split("/")
        if len(parts) != 2:
            raise WrongRepoFormatError("repo needs to be in the format <repo-owner>/<repo-name>")
        return parts[0], parts[1]
