import setuptools

setuptools.setup(
    name="AndroidMalGAN",
    version="0.0.1",
    url="https://github.com/chrome-dino/AndroidMalGAN",
    author="Kenji Harada",
    author_email="kenji.harada@trojans.dsu.edu",
    package_dir={"": "AndroidMalGAN"},
    packages=setuptools.find_namespace_packages(where="AndroidMalGAN"),
    install_requires=['numpy', 'torch', 'matplotlib', 'matplotlib_inline', 'scikit-learn', 'pandas'],
    include_package_data=True
)