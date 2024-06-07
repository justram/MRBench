from setuptools import setup, find_packages

setup(
    name="camera",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "torchvision", 
        "numpy",
        "datasets",
        "hydra-core",
        "omegaconf",
        "datasets", 
        "ranx",
        "transformers",
        "psutil",
    ],
    entry_points={
        'console_scripts': [
            'camera=src.main:main',
        ],
    },
)
