import logging
import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Tuple, Callable, Optional, List, Union

import rich.progress

from dagshub.auth import get_token
from dagshub.auth.token_auth import HTTPBearerAuth
from dagshub.common import config
from dagshub.common.helpers import http_request
from dagshub.common.rich_util import get_rich_progress

logger = logging.getLogger(__name__)

DownloadFunctionType = Callable[[str, Path], None]


def _dagshub_download(url: str, location: Path, auth: HTTPBearerAuth, skip_if_exists: bool):
    logger.debug(f"Download {url} to {location}")

    if skip_if_exists and os.path.exists(location):
        return

    resp = http_request("GET", url, auth=auth, timeout=600)
    try:
        assert resp.status_code == 200
    except AssertionError:
        logger.warning(
            f"Couldn't download file at URL {url}. Response code {resp.status_code} (Body: {resp.content})")
        return
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "wb") as f:
        f.write(resp.content)


def download_files(files: List[Tuple[str, Union[str, Path]]],
                   download_fn: Optional[DownloadFunctionType] = None,
                   threads=32, skip_if_exists=True):
    """
    Download files using multithreading

    Parameters:
        files: list of (download_url: str, file_location: str or Path)
        download_fn: Optional function that will download the file. Needs to receive the two arguments
            If function is not specified, then a default function that downloads a file with DagsHub credentials is used
            CAUTION: function needs to be pickleable since we're using ThreadPool to execute
        threads: number of threads to run this function on
        skip_if_exists: for the default downloader - skip the download if the file exists
    """

    # Convert string paths to Path objects
    for i, file_tuple in enumerate(files):
        if type(file_tuple[1]) is str:
            files[i] = (file_tuple[0], Path(file_tuple[1]))

    if download_fn is None:
        token = config.token or get_token(host=config.host)
        auth = HTTPBearerAuth(token=token)
        download_fn = partial(_dagshub_download, auth=auth, skip_if_exists=skip_if_exists)

    progress = get_rich_progress(rich.progress.MofNCompleteColumn(), transient=False)
    task = progress.add_task("Downloading files...", total=len(files))

    with progress:
        with ThreadPoolExecutor(max_workers=threads) as tp:
            futures = [tp.submit(download_fn, url, location) for (url, location) in files]
            for f in as_completed(futures):
                exc = f.exception()
                if exc is not None:
                    logger.warning(f"Got exception {type(exc)} while downloading file: {exc}")
                progress.update(task, advance=1)
