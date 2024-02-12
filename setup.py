from setuptools import setup, find_packages

setup(
    name="damask",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "pystache",
        "tabulate",
        "litellm",
        "openai",
        # any other dependencies
    ],
    # other metadata
)
