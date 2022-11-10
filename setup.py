import pathlib
import setuptools


here = pathlib.Path(__file__).parent.resolve()
readme = (here / 'README.md').read_text(encoding='utf-8')

# did not include torch and pyscf here
install_requires=['numpy', 'paramiko', 'ruamel.yaml', 'pytest-shutil', 'pydflow']


setuptools.setup(
    name="deepks2",
    version="1.2",
    setup_requires=['setuptools_scm'],
    author="Yifan Shan",
    author_email="xiaoshan@mail.ustc.edu.cn",
    description="DeePKS-flow: a 'dflow' workflow for DeePKS-kit",
    long_description=readme,
    packages=[
        "deepks2",
        "deepks2/flow",
        "deepks2/op/iter_op",
        "deepks2/step/iter_step",
        "deepks2/utils",
    ],
    long_description_content_type="text/markdown",
    # packages=setuptools.find_packages(include=['deepks2', 'deepks2.*']),
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    keywords='deepks2 DeePKS-flow',
    install_requires=install_requires,
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'deepks2=deepks2.main:main_cli',
            'dks2=deepks2.main:main_cli',
        ],
    },
)