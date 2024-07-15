
from setuptools import setup, find_packages

PROJNAME = "uavsar-lidar-project"
DESCRIPTION = "A project that uses NASA JPL UAVSAR data to predict total snow depth."
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
MAINTAINER = "Ibrahim Olalekan Alabi"
MAINTAINER_EMAIL = "ibrahimolalekana@u.boisestate.edu"
URL = "https://github.com/Ibrahim-Ola/uavsar-lidar-project.git"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/Ibrahim-Ola/uavsar-lidar-project/issues",
    "Documentation": "https://github.com/Ibrahim-Ola/uavsar-lidar-project/blob/main/README.md",
    "Source Code": "https://github.com/Ibrahim-Ola/uavsar-lidar-project",
}
VERSION = "0.0.1"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.9"


def setup_package():
    metadata = dict(
        name=PROJNAME,
        version=VERSION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        url=URL,
        project_urls=PROJECT_URLS,
        license=LICENSE,
        python_requires=PYTHON_REQUIRES,
        packages=find_packages(),
        install_requires=[
            "numpy==1.24.4",
            "pandas==2.1.4",
            "matplotlib==3.8.0",
            "seaborn==0.13.1",
            "shap==0.46.0",
            "scikit-learn==1.3.2",
            "xgboost==2.1.0",
            "torch==2.0.1",
            "geopandas==0.14.0",
            "scipy==1.11.2"
        ],
        classifiers=[
            "Development Status :: Mature",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Hydrology",
        ],
        keywords=[
            "UAVSAR",
            "LIDAR",
            "NISAR",
            "Snow",
            "Snow Depth",
            "Snow Water Equivalent",
            "Machine Learning",
            "Remote Sensing",
        ],
    )

    setup(**metadata)

if __name__ == "__main__":
    setup_package()
    print("Setup Complete.")