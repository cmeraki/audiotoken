"""
Setup file for audiotoken
"""

from setuptools import setup, find_packages

setup(
    name="audiotoken",
    version="0.2.1",
    packages=find_packages(),
    description="A package for creating audio tokens",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Romit Jain",
    author_email="romit@merakilabs.com",
    url="https://github.com/cmeraki/audiotoken",
    install_requires=[
        open("requirements.txt").read().splitlines(),
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    entry_points={
        
        "console_scripts": [
            "audiotoken=audiotoken.scrc:main",
        ],
    },
)
