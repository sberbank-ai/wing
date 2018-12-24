from setuptools import setup, find_packages

setup(
    name="wing",
    version='0.1.3',
    description="Weight Of Evidence transformer",
    long_description='',
    author='Trusov Ivan, Cherepanov Yaroslav',
    author_email="YaACherepanov@sberbank.ru",
    url='https://stash.ca.sbrf.ru/projects/CARISKVALID/repos/pyilyabinning/browse',
    license='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False, install_requires=['numpy', 'pandas', 'scikit-learn']
)
