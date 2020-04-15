# coding=utf-8
# Copyright 2020 Konstantin Ustyuzhanin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for Nosferatu.

This script will install Nosferatu as a Python module.

See: https://github.com/ustyuzhaninky/nosferatu

"""

from os import path
from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

install_requires = ['gin-config >= 0.1.1', 'absl-py >= 0.2.2',
                    'opencv-python >= 3.4.1.15',
                    'gym >= 0.10.5', 'Pillow >= 5.4.1']
tests_require = ['gin-config >= 0.1.1', 'absl-py >= 0.2.2',
                 'opencv-python >= 3.4.1.15',
                 'gym >= 0.10.5', 'mock >= 1.0.0', 'Pillow >= 5.4.1']

nosferatu_description = (
    'Nosferatu: A dopamine-based DQN with exploratory traits')

setup(
    name='nosferatu',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    package_data={'testdata': ['testdata/*.gin']},
    install_requires=install_requires,
    tests_require=tests_require,
    description=nosferatu_description,
    long_description=nosferatu_description,
    url='https://github.com/ustyuzhaninky/nosferatu',  # Optional
    author='Konstantin Ustyuzhanin',  # Optional
    classifiers=[  # Optional
        'Development Status :: 0 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    project_urls={  # Optional
        'Documentation': 'https://github.com/ustyuzhaninky/nosferatu',
        'Bug Reports': 'https://github.com/ustyuzhaninky/nosferatu/issues',
        'Source': 'https://github.com/ustyuzhaninky/nosferatu',
    },
    license='Apache 2.0',
    keywords='capsules exploratory dopamine-rl reinforcement-learning python machine learning'
)
