# Create a setup.py file to install the package
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    req_packs = f.read().splitlines()

setup(
    name="brainvolcnn",
    version="0.0.1",
    description="Brain Volume CNN",
    url="https://github.com/eminSerin/brainvolcnn",
    author="Emin Serin",
    author_email="emin.serin@charite.de",
    license="GNU General Public License v3.0",
    packages=find_packages(),
    install_requires=req_packs,
    zip_safe=False,
)
