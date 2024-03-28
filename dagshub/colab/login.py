from dagshub.auth import add_oauth_token, get_token
from dagshub.auth.tokens import get_user_of_token
from dagshub.common.api import RepoAPI
from dagshub.common.api.repo import RepoNotFoundError
from dagshub.common.helpers import log_message
from dagshub.upload import create_repo

COLAB_REPO_NAME = "dagshub-drive"


def login():
    """
    Run custom colab-specific flow, which helps users with setting up a repository,
    storage of which will be used as an alternative to Google Drive
    """
    try:
        get_token(fail_if_no_token=True)
    except RuntimeError:
        add_oauth_token(referrer="colab")

    token = get_token()
    username = get_user_of_token(token)

    colab_repo = RepoAPI(f"{username}/{COLAB_REPO_NAME}")
    try:
        colab_repo.get_repo_info()
    except RepoNotFoundError:
        create_repo(COLAB_REPO_NAME)
    log_message(f"Repository {colab_repo.full_name} is ready for use with Colab. Link to the repository:")
    print(colab_repo.repo_url)
