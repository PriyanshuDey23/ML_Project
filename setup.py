# find Packages :- Find All the packages
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .' # in Requirements

def get_requirements(file_path:str)->List[str]: 
    """
        This Function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements] #Replace / with space

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT) # This should not be considered as requirements
    
    return requirements        



setup(
name='ML_Project',
version='0.0.1',
author="Priyanshu Dey",
author_email="priyanshudey.ds@gmail.com",
packages=find_packages(), # How many folder will the (_init_) file, that will be considered as package
install_requires=get_requirements('requirements.txt') # All The Requirements

)