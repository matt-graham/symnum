setuptools.setup(
    name='symnum',
    version='0.1.0',
    author='Matt Graham',
    description=(
        'Symbolically construct NumPy functions and their derivatives'
    ),
    url='https://github.com/matt-graham/symnum.git',
    packages=['symnum'],
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
    license_file='LICENSE',
    install_requires=['numpy>=1.15', 'sympy>=1.5'],
    python_requires='>=3.6',
)