
from setuptools import setup, find_packages

setup(
    name='spectra',
    version='0.1.0',
    packages=find_packages(),
    data_files={"spectra/data": ["data/*"]},
    install_requires = [
        'numpy>=1.20.3',
        'scipy>=1.7.3',
        'scanpy>=1.8.2',
        'torch>=1.10.1',
        'opt-einsum>=3.3.0',
        'pandas>=1.3.5',
        'tqdm>=4.62.3',
        'pyvis>=0.1.9']
        
)
