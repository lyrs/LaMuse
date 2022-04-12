import setuptools

#from LaMuse.Musesetup import version_number

with open("LaMuse/version_number") as vn:
   version_number = vn.readline()

with open("LaMuse/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LaMuse", 
    version=version_number,
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
    package_data={'LaMuse': ['mask_rcnn_coco.h5', 'BaseImages_objets/*/*png', 'Paintings/*', 'Watermark.png', 'version_number', 'README.md']},
)
