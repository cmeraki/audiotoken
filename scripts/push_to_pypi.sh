#!/bin/bash

# Ensure the script exits on any error
set -e

# Update version (optional)
# Uncomment and modify if you want to automatically update the version
# sed -i 's/version=.*/version="1.0.1",/' setup.py

# Clean up any old distributions
rm -rf dist build *.egg-info

# Install or upgrade build tools
pip install --upgrade setuptools wheel twine

# Build the package
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*

echo "Package built and uploaded successfully!"
