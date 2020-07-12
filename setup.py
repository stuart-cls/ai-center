from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='aicenter',
    version='2020.7.1',
    packages=['aicenter'],
    url='',
    license='MIT',
    author='Michel Fodje',
    author_email='michel.fodje@lightsource.ca',
    install_requires=['wheel'] + requirements,
    description='IOC Application for Auto Centering using DarkNet',
    scripts=[
        'bin/ai-center-ioc'
    ],
)
