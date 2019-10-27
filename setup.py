from setuptools import setup, find_namespace_packages

setup(
    name='radtree',
    description='Radial Plot for sklearn Decision Trees',
    version='0.0.1',
    url='http://github.com/poctaviano/radtree',
    author='Pedro Octaviano',
    author_email='pedro.octaviano@gmail.com',
    license='MIT',
    # scripts=['radtree'],
    # packages=['radtree.py'], find_namespace_packages(include=['mynamespace.*'])
    packages=['radtree'],
    zip_safe=False
)
