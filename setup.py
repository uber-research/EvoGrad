# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evograd",
    version="0.1.2",
    author="Alex Gajewski",
    author_email="agajews@gmail.com",
    description="A lightweight tool for differentiating through expectations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uber-research/EvoGrad",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "torch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
