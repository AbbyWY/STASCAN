from setuptools import Command, find_packages, setup

__lib_name__ = "STASCAN"
__lib_version__ = "1.0.0"
__description__ = "An AI-driven method for enhanced cellular organizational map in spatial transcriptomics"
__url__ = "https://github.com/yangteam/wuying/STASCAN"
__author__ = "Ying Wu"
__author_email__ = "wuy@big.ac.cn"
__license__ = "MIT"
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['STASCAN'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)
