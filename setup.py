from setuptools import find_packages, setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Code to Google AI4Code competition on kaggle.com',
    long_description=long_description,
    author='Dmitry Sokolevski',
    license='MIT',
)