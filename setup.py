#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for FinOps Cloud Cost Forecasting package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="finops-ev-telematics",
    version="1.0.0",
    author="Víctor Valdivieso, Francisco Manuel Arrabal-Campos",
    author_email="your-email@university.edu",
    description="Machine Learning-Powered FinOps for EV Telematics Cloud Cost Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/finops-ev-telematics",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/finops-ev-telematics/issues",
        "Documentation": "https://github.com/your-username/finops-ev-telematics#readme",
        "Source Code": "https://github.com/your-username/finops-ev-telematics",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
    ],
    keywords=[
        "finops",
        "cloud-cost-optimization",
        "machine-learning",
        "forecasting",
        "electric-vehicles",
        "telematics",
        "lstm",
        "arima",
        "prophet",
        "time-series",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "finops-forecast=finops_forecasting:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
