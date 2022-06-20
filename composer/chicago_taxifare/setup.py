from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='chicago_taxifare',
    version='1.0',
    author='owners name',
    author_email='user_name@email-address.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='',
    requires=[]
)
