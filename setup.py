from setuptools import setup, find_packages

setup(
    name="optical-compression",
    version="0.1.0",
    author="Optical Compression Team",
    description="Optical compression simulation framework for replacing H.265/JPEG",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
    ],
)