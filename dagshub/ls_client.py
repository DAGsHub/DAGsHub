from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
from dagshub.data_engine.model.errors import LSInitializingError
from contextlib import _GeneratorContextManager
from dagshub.common.util import lazy_load
from json import JSONDecodeError
from typing import Optional
from itertools import tee
import importlib.util
import semver
import types

from dagshub.common.api.repo import RepoAPI
from dagshub.auth import get_token
from dagshub.common import config


ls_sdk = lazy_load("label_studio_sdk")


class _TenaciousLSCLientWrapper:
    def __init__(self, func):
        self.func = func

    @retry(
        retry=retry_if_exception_type((LSInitializingError, JSONDecodeError, ls_sdk.core.ApiError)),
        wait=wait_fixed(3),
        stop=stop_after_attempt(5),
    )
    def wrapped_func(self, *args, **kwargs):
        res = self.func(*args, **kwargs)

        if isinstance(res, types.GeneratorType):
            proxy, res = tee(res)
            if next(proxy).startswith(b"<!DOCTYPE html>"):
                raise LSInitializingError()
        elif isinstance(res, _GeneratorContextManager):
            return res
        elif isinstance(res, bytes):
            if res.startswith("<!DOCTYPE html>"):
                raise LSInitializingError()
        else:
            if res.text.startswith("<!DOCTYPE html>"):
                raise LSInitializingError()
            elif res.status_code // 100 != 2:
                raise RuntimeError(f"Process failed! Server Response: {res.text}")
        return res


def _use_legacy_client():
    """
    https://github.com/HumanSignal/label-studio/releases/tag/1.13.0, \
            https://github.com/HumanSignal/label-studio/pull/5961; \
            introduces breaking changes; anyone using SDK < 1.0 should use the legacy client.
    :meta experimental:
    """
    import label_studio_sdk

    return semver.compare("1.0.0", label_studio_sdk.__version__) == 1


@retry(retry=retry_if_exception_type(JSONDecodeError), wait=wait_fixed(3), stop=stop_after_attempt(5))
def get_label_studio_client(
    repo: str, legacy_client: Optional[bool] = None, host: Optional[str] = None, token: Optional[str] = None
):
    """
    Creates a
    `label_studio_sdk.client.LabelStudio <https://api.labelstud.io/api-reference/introduction/getting-started>`_ /
    `label_studio_sdk.Client (legacy) <https://labelstud.io/guide/sdk>`_
    object to interact with the LabelStudio instance associated with the repository.

    Args:
        repo: Name of the repo in the format of ``username/reponame``
        legacy_client: if True, returns the older legacy LabelStudio Client.
        host: URL of the hosted DagsHub instance. default is ``https://dagshub.com``.
        token: (optional, default: None) use this token for LS requests. By default, will use current user's token.

    Returns:
        `label_studio_sdk.Client` / `label_studio_sdk.client.LabelStudio` object
    """

    if importlib.util.find_spec("label_studio_sdk") is None:
        raise ModuleNotFoundError("Could not import module label_studio_sdk. Make sure to pip install label_studio_sdk")

    if legacy_client is None:
        legacy_client = _use_legacy_client()

    if not host:
        host = config.host

    if legacy_client:
        from label_studio_sdk import Client as LabelStudio
    else:
        from label_studio_sdk.client import LabelStudio

    repo_api = RepoAPI(repo, host=host)
    kwargs = {
        "url" if legacy_client else "base_url": repo_api.label_studio_api_url()[:-4],
        "api_key": token if token is not None else get_token(host=host),
    }

    ls_client = LabelStudio(**kwargs)
    if legacy_client:
        ls_client.make_request = _TenaciousLSCLientWrapper(ls_client.make_request).wrapped_func
    else:
        ls_client._client_wrapper.httpx_client.request = _TenaciousLSCLientWrapper(
            ls_client._client_wrapper.httpx_client.request
        ).wrapped_func
        ls_client.projects.exports.create_export = _TenaciousLSCLientWrapper(
            ls_client.projects.exports.create_export
        ).wrapped_func

    return ls_client
