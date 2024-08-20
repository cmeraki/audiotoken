"""
Setup file for audiotoken
"""

from setuptools import setup, find_packages

setup(
    name="audiotoken",
    version="0.3.0",
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
    dependency_links=[
        # Make sure to include the `#egg` portion so the `install_requires` recognizes the package
        'git+https://github.com/suno-ai/bark.git#egg=bark',
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
