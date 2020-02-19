from setuptools import setup, find_packages

setup(
    name="wing",
    version='0.1.5',
    description="Weight Of Evidence transformer",
    long_description='',
    author='Trusov Ivan, Cherepanov Yaroslav',
    author_email="YaACherepanov@sberbank.ru",
    url='https://github.com/sberbank-ai/wing',
    license='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False, install_requires=['numpy', 'pandas', 'scikit-learn']
)
