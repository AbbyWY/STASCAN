from setuptools import Command, find_packages, setup

__lib_name__ = "STASCAN"
__lib_version__ = "1.1.0"
__description__ = "An AI-driven method for enhanced cellular organizational map in spatial transcriptomics"
__url__ = "https://github.com/AbbyWY/STASCAN"
__author__ = "Ying Wu"
__author_email__ = "wuy@big.ac.cn"
__license__ = "MIT"
__requires__ = ["requests",]
__long_description__ = "Here we propose STASCAN, which both gene expression and the morphological information are simultaneously utilized to improve the cellular resolution of captured domains and even gap regions. Besides, STASCAN is further designed to enable cell-type predictions at subdivide-spot resolution and construction of the 3D spatial cell map from histology images alone."


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
