import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LaMuse", 
    version="0.1.0",
    author="Bart Lamiroy",
    author_email="Bart.Lamiroy@univ-reims.fr",
    description="LaMuse, deep learning for painters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lamiroy/LaMuse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'LaMuse': ['mask_rcnn_coco.h5', 'BaseImages_objets/*/*png', 'Paintings/*', 'Watermark.png']},
)
