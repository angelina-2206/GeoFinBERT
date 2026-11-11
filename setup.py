from setuptools import setup, find_packages

setup(
    name="geofinbert",
    version="1.0.0",
    description="GeoFinBERT: Adapting FinBERT for Geopolitical and Energy Market Sentiment Analysis",
    author="Academic ML Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
)

