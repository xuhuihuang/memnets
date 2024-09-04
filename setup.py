from setuptools import setup, find_packages
import re

install_requires = ["numpy<=1.26","scipy","torch","tqdm"]
_extras_require = ["matplotlib","scikit-learn","jupyter"]
extras_require = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _extras_require)}

CLASSIFIERS = """\
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Scientific/Engineering
Topic :: Artificial Intelligence
"""

setup(
    name ='memnets',
    version = '1.0.0',
    python_requires = '>=3.9.0',
    install_requires = install_requires,
    extras_require = extras_require,
    description = 'MEMnets for identifying CVs of biomolecular conformational dynamics',
    long_description = open("README.md", "r", encoding="utf-8").read(),
    license = 'MIT',
    author = 'Bojun Liu',
    author_email = 'bliu293@wisc.edu',
    packages = find_packages(),
    classifiers = CLASSIFIERS.splitlines()
)
