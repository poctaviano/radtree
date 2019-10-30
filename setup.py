from setuptools import setup

setup(
    name='radtree',
    description='Radial Plot for sklearn Decision Trees',
    version='0.0.3',
    url='http://github.com/poctaviano/radtree',
    author='Pedro Octaviano',
    author_email='pedrooctaviano@gmail.com',
    license='MIT',
    packages=['radtree'],
    install_requires=['networkx','tqdm', 'sklearn', 'matplotlib', 'scipy'],
    zip_safe=False
)
