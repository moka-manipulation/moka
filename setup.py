from setuptools import setup, find_packages

setup(
    name='cvp',
    version='0.1.1',
    packages=find_packages(),
    description='control with visual prompts',
    url='https://github.com/kuanfang/cvp',
    author='auth',
    author_email='fangchen_liu@berkeley.edu',
    license='MIT',
    install_requires=[
        'typing',
        'typing_extensions',
    ],
    zip_safe=False
)
