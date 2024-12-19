from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
    
with open('requirements.txt', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setup(
    name='distvae_tabular',
    version='0.2.4',
    author='Seunghwan An',
    author_email='dpeltms79@gmail.com',
    description='Package for Synthetic Data Generation using Distributional Learninig of VAE',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/an-seunghwan/DistVAE-Tabular',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.11',
    install_requires=install_requires
)