from setuptools import setup, find_namespace_packages

setup(
    name='radial-tree',
    description='Radial Plot for sklearn Decision Trees',
    version='0.0.1',
    url='git@github.com:poctaviano/radtree.git',
    author='Pedro Octaviano',
    author_email='pedro.octaviano@gmail.com',
    license='MIT License',
    # scripts=['radtree'],
    # packages=['radtree.py'], find_namespace_packages(include=['mynamespace.*'])
    packages=find_namespace_packages(include=['radtree.*']),
    zip_safe=False
)
