# setup.py
from setuptools import setup, find_packages

# read requirements
with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]


setup(
    name='yolo_3d_track',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
)
