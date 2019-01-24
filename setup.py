from setuptools import setup, find_packages

version = {}
with open('psluncert/version.py', 'r') as f:
    exec(f.read(), version)

setup(
    name='psluncert',
    version=version['__version__'],
    packages=find_packages(),
    description='PSL Uncertainty Calculator',
    author='Collin J. Delker',
    author_email='cjdelke@sandia.gov',
    install_requires=[
        'sympy>=1.3',
        'scipy>=1.1',
        'numpy>=1.15',
        ],
    entry_points={
        'console_scripts': ['psluncert = psluncert.__main__:main_unc',
                            'psluncertf = psluncert.__main__:main_setup',
                            'psluncertrev = psluncert.__main__:main_reverse',
                            'psluncertrisk = psluncert.__main__:main_risk',
                            'psluncertfit = psluncert.__main__:main_curvefit',
                            ],
        'gui_scripts': ['psluncertui = psluncert.gui.gui_main:main'],
        },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        ]
    )
