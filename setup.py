import setuptools

requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
        if req.startswith('git+'):
            pkg_name = req.split('/')[-1].replace('.git', '')
            if "#egg=" in pkg_name:
                pkg_name = pkg_name.split("#egg=")[1]
            requirements.append(f'{pkg_name} @ {req}')
        else:
            requirements.append(req)

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_version(rel_path):
    for line in open(rel_path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="pyterrier-dr",
    version=get_version('pyterrier_dr/__init__.py'),
    author="Sean MacAvaney",
    author_email='sean.macavaney@glasgow.ac.uk',
    description="PyTerrier components for dense retrieval",
    long_description=long_description,
    url='https://github.com/terrierteam/pyterrier_dr',
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require={
        'bgem3': ['FlagEmbedding'],
    },
    python_requires='>=3.6',
    entry_points={
        'pyterrier.artifact': [
            'dense_index.flex = pyterrier_dr:FlexIndex',
            'cde_cache.np_pickle = pyterrier_dr:CDECache',
        ],
    },
)
