"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

from setuptools import setup, find_packages

setup(
    name='agri-count', # name of the package
    version='1.0.0', # version of the package
    description='Calculation of peanut seeds within single video frame',
    author='tvisi', # author of the package
    author_email='tvisi.net@gmail.com', # package author mail
    url='', # url of package repository
    packages=find_packages(), # automatically find packages in 'src' directory
    install_requires= [
        'torch',
        'torchvision',
        'opencv-python',
        'matplotlib',
        'numpy',
        'lightning',
        'torchinfo',
        'git+https://github.com/facebookresearch/segment-anything.git'
    ], # list of the dependencies required by the package
    classifiers=[
        'Programming Language :: Python :: 3.12.3'
    ] # list of classifiers describing package
)