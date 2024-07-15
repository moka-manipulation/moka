from setuptools import setup, find_packages

setup(
    name='moka',
    version='0.1',
    packages=find_packages(),
    description='marking open-world keypoint affordances',
    url='git@github.com:moka-manipulation/moka.git',
    author='auth',
    author_email='fangchen_liu@berkeley.edu, kuanfang@cornell.edu',
    license='MIT',
    install_requires=[
        'typing',
        'typing_extensions',
    ],
    zip_safe=False
)
