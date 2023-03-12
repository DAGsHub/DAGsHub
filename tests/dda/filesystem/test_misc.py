import os.path

import pytest
from pathlib import Path
from dagshub.streaming.dataclasses import DagshubPath, DagshubPathType


@pytest.mark.parametrize(
    "path,expected",
    [
        ("regular/path/in/repo", False),
        (".", False),
        (".git/", True),
        (".git/and/then/some", True),
        (".dvc/", True),
        (".dvc/and/then/some", True),
        ("repo/file.dvc/", False),
        ("repo/file.git/", False),
        ("venv/lib/site-packages/some-package", True),
    ],
)
def test_passthrough_path(path, expected):
    path = DagshubPath(Path(os.path.abspath(path)), Path(path))
    actual = DagshubPathType.PASSTHROUGH_PATH in path.path_type
    assert actual == expected
