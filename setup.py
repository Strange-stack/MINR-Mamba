from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    requirements_path = Path(__file__).resolve().parent / 'requirements.txt'
    if not requirements_path.exists():
        return []
    return [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]


setup(
    name='minr-mamba',
    version='0.1.0',
    description='MINR-Mamba image deraining model',
    packages=find_packages(exclude=('experiments', 'results', 'tb_logger')),
    install_requires=read_requirements(),
    zip_safe=False,
)
