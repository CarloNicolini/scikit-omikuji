from setuptools import find_packages, setup
from os import path

def load_readme():
    curr_dir = path.abspath(path.dirname(__file__))
    with open(path.join(curr_dir, "README.md"), encoding="utf-8") as f:
        return f.read()

setup(
    name='skomikuji',
    packages=find_packages(),
    version='0.2.0',
    description='A Python wrapper supporting sparse input matrices for multilabel classification in extreme settings.',
    author='Carlo Nicolini',
    author_email="c.nicolini@ipazia.com",
    install_requires=["cython","numba","numpy","scikit-learn"],
    license='MIT',
)
