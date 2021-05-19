import os
from setuptools import find_packages, setup


__version__ = "0.1"

if "VERSION" in os.environ:
    BUILD_NUMBER = os.environ["VERSION"].rsplit(".", 1)[-1]
else:
    BUILD_NUMBER = os.environ.get("BUILD_NUMBER", "dev")

dependencies = [
    "click>=7.0",
    "numpy>=1.17.0,<2.0",
    "opencv-python-headless",
    "tensorboard==2.4",
    "scikit-learn>=0.21.0,<1.0",
    "scikit-image",
    "git-python>=1.0.3",
    "tensorflow==2.4.1",
    "xlsxwriter>=1.2.9",
    "tensorflow-addons==0.8.1",
]

setup(
    name="localize_pets",
    version="{0}.{1}".format(__version__, BUILD_NUMBER),
    description="A package to localize pets",
    author="Deepan Chakravarthi Padmanabhan",
    install_requires=dependencies,
    packages=find_packages(),
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            "localize_pets_train=localize_pets.train:train",
            "localize_pets_evaluate=localize_pets.evaluate.evaluate:evaluate",
        ]
    ),
    python_requires=">=3.6,<=3.9",
)
