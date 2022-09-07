from io import open
from setuptools import find_packages, setup

with open("requirements.txt") as fp:
    install_requires = fp.read().split('\n')

with open('tests/test_requirements.txt') as fp:
    tests_require = fp.read()

setup(
    name="digital_eval",
    author="Uwe Hartwig",
    author_email="development@bibliothek.uni-halle.de",
    description="Evaluate Mass Digitalization Data",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    version='1.0.0',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=install_requires,
    tests_require=tests_require,
    entry_points={
        'console_scripts': [
            'digital-eval=digital_eval.cli:main',
        ]
    },
)
