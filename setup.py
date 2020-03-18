from setuptools import setup

setup(
   name='t3r utilities',
   version='0.1.0',
   author='sponchcafe',
   modules=['t3r_util'],
   license='LICENSE.txt',
   description='Helper utility to work with t3r timetag files.',
   long_description=open('README.md').read(),
   install_requires=[
       "numpy",
   ],
)
