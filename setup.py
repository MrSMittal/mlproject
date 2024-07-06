from setuptools import find_packages,setup
from typing import List


HYPHE_E_DOT="-e."
def get_requirements(file_path:str)->list:
    requirements=[]
    with open(file_path) as fil_obj:
        requirements=fil_obj.readlines()
        requirements= [req.replace("\n","")for req in requirements]
        if HYPHE_E_DOT in requirements:
            requirements.remove(HYPHE_E_DOT)
    return requirements

setup(
    name='ML-Project',
    version="0.0.1",
    author="Shivam",
    author_email="mittalshivam40@gmail.com",
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt'),


)