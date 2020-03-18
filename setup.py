from setuptools import setup

setup(
   name='t3r_utils',
   version='0.1.0',
   author='sponchcafe',
   py_modules=['t3r_utils'],
   license='LICENSE.txt',
   description='Helper utility to work with t3r timetag files.',
   long_description=open('README.md').read(),
   install_requires=[
       "numpy",
   ],
)
