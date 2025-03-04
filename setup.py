from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='LDP_hierarchical_mechanism',
    version='0.1.3',
    description='Local Differential Privacy for Range Queries',
    long_description=long_description,  # Include README content here
    long_description_content_type='text/markdown',  # Specify Markdown format
    author='Fabrizio Boninsegna',
    url='https://github.com/NynsenFaber/LDP_hierarchical_mechanism',
    packages=find_packages(),
    install_requires=[
        'pure-ldp>=1.2.0',
        'scikit-learn==1.6.1',
        'statsmodels==0.14.4'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)