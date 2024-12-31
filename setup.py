"""Setup file for tokenizers."""

import pathlib

from setuptools import find_packages, setup

setup(
    name="simple_tokenizers",
    version=pathlib.Path("simple_tokenizers/VERSION").read_text(encoding="utf-8"),
    description="Simple tokenizers is a collection of tokenization implementations focused on transparency and readability.",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/dakofler/simple_tokenizers/",
    author="Daniel Kofler",
    author_email="dkofler@outlook.com",
    license="MIT",
    project_urls={
        "Source Code": "https://github.com/dakofler/simple_tokenizers",
        "Issues": "https://github.com/dakofler/simple_tokenizers/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=[
        "regex>=2023.12.25",
        "tqdm>=4.66.2",
    ],
    extras_require={
        "dev": [
            "ipywidgets>=8.1.5",
            "mypy>=1.11.2",
            "twine>=5.1.1",
            "wheel>=0.43.0",
        ]
    },
    packages=find_packages(exclude=["tests", ".github", ".venv", "docs"]),
    include_package_data=True,
)
