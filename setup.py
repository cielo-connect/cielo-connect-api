from setuptools import setup, find_packages

setup(
    name="cielo-connect-api",
    version="1.0.2",
    description="Python package for Cielo Connect APIs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="Cielo Wigle Inc. and Muhammad Ihsan",
    packages=find_packages(where=".", include=["cieloconnectapi"]),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    package_data={
        "": ["README.md", "LICENSE"],
    },
    include_package_data=True,
)

