import os
from dagshub.streaming import install_hooks


def test_torch_load():
    install_hooks(project_root="/Users/simonlousky/workspace/dags/user_repos/public/SavtaDepth")
    res = list(os.walk("src/data/processed"))
    print(res)
    # from fastai.vision.all import get_files
    # files = get_files("src/data/processed/train", extensions=".jpg")
    # print(files)



test_torch_load()
