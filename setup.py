from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
    
with open('requirements.txt', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setup(
    name='distvae_tabular',
    version='0.0.10',
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
    install_requires=install_requires
)