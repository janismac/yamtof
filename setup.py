from distutils.core import setup
setup(
    name = 'yamtof',
    version = '0.1',
    license = 'apache-2.0',
    packages = ['yamtof'],
    description = 'Yamtof provides a simple and convenient interface for writing multi-phase optimal control problems.',
    author = 'Janis Maczijewski',
    author_email = 'Janis.Maczijewski@rwth-aachen.de',
    url = 'https://github.com/janismac/yamtof',
    download_url = 'https://github.com/janismac/yamtof/archive/v0.1.tar.gz',
    install_requires = ['casadi','numpy','matplotlib'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
