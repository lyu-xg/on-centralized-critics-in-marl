#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

setup(
    name='pg_marl',
    version='0.0.1',
    description='pg_marl - policy gradient for multi-agent reinforcement learning',
    packages=['pg_marl'],
    package_dir={'': 'src'},

    scripts=[
        'scripts/maac_cenV_rnn.py',
        'scripts/maac_cenV_agrnn.py',
        'scripts/maac_cenV_shc.py',
    ],

    license='MIT',
)
