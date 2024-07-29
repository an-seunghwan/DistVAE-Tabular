from setuptools import setup, find_packages
import re

# Read the version from the __init__.py file
def read_version():
    with open("distvae_tabular/__init__.py", "r") as f:
        version_file = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

with open('README.md', encoding='utf-8') as f:
    long_description = f.read(),

setup(
    name='distvae_tabular',
    version=read_version(),
    author='Seunghwan An',
    author_email='dpeltms79@gmail.com',
    description='DistVAE Implementation Package for Synthetic Data Generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/an-seunghwan/DistVAE-Tabular',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.5.1',
        'scipy==1.14.0',
        'torch==2.2.2',
        'tqdm==4.66.4',
    ]
)