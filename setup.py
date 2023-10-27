#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:45:10 2023

@author: vincent
"""

from setuptools import setup

setup(
    name='cst-rautils',
    version='0.1',
    packages=['cst-rautils'],
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'
        ],
)

