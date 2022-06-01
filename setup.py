from setuptools import find_packages, setup

setup(
    name='muse',
    packages=find_packages(),
    install_requires=[
        'pyroomacoustics',
        'librosa',
        'seaborn',
        'matplotlib',
        'numpy',
        'scipy',
        'notebook'
    ]
)
