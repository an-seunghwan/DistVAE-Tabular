from setuptools import setup, find_packages

setup(
    name='distvae-tabular',
    version='0.0.1',
    author='Seunghwan An',
    author_email='dpeltms79@gmail.com',
    description='DistVAE Implementation Package for Synthetic Data Generation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/an-seunghwan/DistVAE-Tabular',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)