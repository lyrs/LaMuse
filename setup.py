import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lamiroy", # Replace with your own username
    version="0.0.1",
    author="Bart Lamiroy",
    author_email="Bart.Lamiroy@univ-reims.fr",
    description="LaMuse, deep learning for painters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lamiroy/LaMuse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)