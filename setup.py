import os

# import project package
import src

from setuptools import setup, find_packages

def readme() -> str:
    """Utility function to read the README.md.

    Used for the `long_description`. It's nice, because now
    1) we have a top level README file and
    2) it's easier to type in the README file than to put a raw string in below.

    Args:
        nothing

    Returns:
        String of readed README.md file.
    """
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='gp23package',
    version='0.1',
    author='Tomasz G.',
    author_email='tgprogramming1@gmail.com',
    description='Guided Project 23 from DataQuest.io Data Scientist in Python path. Predicting house sale prices based on Ames, Iowa dataset.',
    python_requires='>=3.9',
    license='MIT',
    url='',
    packages=find_packages(),
    long_description=readme(),
)
