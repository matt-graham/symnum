import setuptools

setuptools.setup(
    name='symnum',
    version='0.1.2',
    author='Matt Graham',
    description='Symbolically construct NumPy functions and their derivatives',
    long_description=(
        'SymNum is a Python package that acts a bridge between NumPy and SymPy,'
        ' providing a NumPy-like interface that can be used to symbolically '
        'define functions which take arrays as arguments and return arrays or '
        'scalars as values. A series of Autograd style functional differential '
        'operators are also provided to construct derivatives of symbolic '
        'functions, with the option to generate NumPy code to numerically '
        'evaluate these derivative functions.'
    ),
    url='https://github.com/matt-graham/symnum.git',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers'
    ],
    keywords='sympy numpy differentiation',
    license='MIT',
    license_files=('LICENSE',),
    install_requires=['numpy>=1.15', 'sympy>=1.8'],
    python_requires='>=3.6',
)
