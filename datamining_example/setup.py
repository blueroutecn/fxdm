from setuptools import setup


setup(
    name="datamining-examples",
    description="Examples for the python-weka-wrapper3 library.",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='GNU General Public License version 3.0 (GPLv3)',
    package_dir={
        '': 'src'
    },
    packages=[
        "wekaexamples",
        "wekaexamples.associations",
        "wekaexamples.attribute_selection",
        "wekaexamples.book",
        "wekaexamples.classifiers",
        "wekaexamples.core"
    ],
    version="0.1.0",
    install_requires=[
        "python-weka-wrapper3>0.1.7",
    ],
)
