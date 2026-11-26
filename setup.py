from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="cs329a_hw3",
    version="0.1.0",
    author="stanford-cs329a",
    author_email="",
    description="A toolbox for evaluating self-improvement techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-cs329a/cs329a-homework3-fall2025-public",
    packages=find_packages(),
    package_data={"cs329a_hw3": ["weather_codes.json"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
