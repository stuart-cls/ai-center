from setuptools import setup

setup(
    name='ai-center',
    version='2020.7.1',
    packages=['aicenter'],
    url='',
    license='MIT',
    author='Michel Fodje',
    author_email='michel.fodje@lightsource.ca',
    description='IOC Application for Auto Centering using DarkNet',
    scripts=[
        'bin/ai-center-ioc'
    ],
)
