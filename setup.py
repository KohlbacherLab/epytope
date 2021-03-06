from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from distutils.core import Extension
from codecs import open  # To use a consistent encoding
from os import path
import glob

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    readme = f.read()

#d2s_src_dir = path.join(path.join('epytope', 'Distance2Self'), 'src')
#d2s_module = Extension('epytope.d2s',
#                       define_macros=[('MAJOR_VERSION', '1'),
#                                      ('MINOR_VERSION', '0')],
#                       include_dirs=[d2s_src_dir],
#                       libraries=['boost_serialization', 'boost_python'],
#                       #library_dirs = ['/usr/local/lib'],
#                       depends=[path.join(d2s_src_dir, 'distance2self.hpp')],
#                       sources=[path.join(d2s_src_dir, 'distance2self.cpp')])


#data_files = list()
# directories = glob.glob('epytope/Data/svms/*/')
# for directory in directories:
#     files = glob.glob(directory + '*')
#     data_files.append((directory, files))
#directories = glob.glob('epytope/Data/examples/')
#for directory in directories:
#    files = glob.glob(directory + '*')
#    data_files.append((directory, files))
#
# d2s_files = glob.glob(d2s_dir + "src/" + '*')
#data_files.append((d2s_dir + "src/", d2s_files))

#for packaging files must be in a package (with init) and listed in package_data
# package-externals can be included with data_files,
# and there is a bug in patternmatching http://bugs.python.org/issue19286
# install unclear for data_files

setup(
    name='epytope',

    # Version:
    version='3.0.0rc1',

    description='A Framework for Epitope Detection and Vaccine Design',
    long_description=readme,

    # The project's main homepage.
    url='https://github.com/KohlbacherLab/epytope',

    # Author details
    author='Benjamin Schubert, Mathias Walzer',
    author_email='schubert@informatik.uni-tuebingen.de, walzer@informatik.uni-tuebingen.de',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Biologists, Pharmacologist, Developer',
        'Topic :: Immunoinformatics :: Prediction Tools',

        # The license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    # What epytope relates to:
    keywords='epitope prediction MHC vaccine',

    # Specify  packages via find_packages() and exclude the tests and 
    # documentation:
    packages=find_packages(),
    #packages=find_packages(exclude=['epytope.test', 'epytope.doc', 'epytope.tutorials']),

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #include_package_data=True,
    package_data={
        'epytope.Data.examples': ['*.*'],
        'epytope.Data.svms.svmtap': ['*'],
        'epytope.Data.svms.svmhc': ['*'],
        'epytope.Data.svms.unitope': ['*'],
        #'epytope.Distance2Self': ['src/*'],  #does not get installed, because the src folder is no package folder - compiles ok
    },

    #package_data is a lie: http://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute

    # 'package_data' is used to also install non package data files
    # see http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # example:
    #data_files=data_files,

    # Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # IMPORTANT: script names need to be in lower case ! ! ! (otherwise 
    # deinstallation does not work)
    #entry_points={
    #    'console_scripts': [
    #        'epitopeprediction=epytope.Apps.EpitopePrediction:main',
    #    ],
    #},

    #ext_modules=[helloworld_module],
    #ext_modules=[d2s_module],

    # Run-time dependencies. (will be installed by pip when epytope is installed)
    # TODO: find alternative for SMVlight scikitlearn
    install_requires=[
            'setuptools>=18.2',
            'pandas',
            'pyomo>=4.0',
            'PyMySQL',
            'biopython',
            'pyVCF',
            'mhcflurry<=1.4.3',
            'mhcnuggets',
            'keras<=2.3.1',         # legacy tensorflow required by mhcnuggets resolves to incompatible keras version
            'h5py<=2.10.0',         # mhcnuggets fails to read model with newer versions
            ],

)
