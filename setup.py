from setuptools import setup

__VERSION_MAJOR = 0
__VERSION_MINOR = 0
__VERSION_PATCH = 1
__VERSION__ = f"{__VERSION_MAJOR}.{__VERSION_MINOR}.{__VERSION_PATCH}"

__PROJECT_NAME = 'pycodelib'
__PACKAGE_NAME = 'pycodelib'

setup(name=__PROJECT_NAME,
      version=__VERSION__,
      description='HDF5 Database for Images',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      keywords=['util'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
      ],
      url='https://github.com/CielAl/pycodelib',
      author='***',
      author_email='**',
      packages=__PACKAGE_NAME,
      install_requires=[
          'tables>=3.4.4',
          'scikit-learn>=0.20.2',
          'scikit-image>=0.14.1',
          'numpy>=1.14.5',
          'tqdm>=4.28.1',
          'lazy_property>=0.0.1',
          'pillow>=5.4.0',
          'joblib>=0.13.2',
          'torch>=1.0.1',
          'opencv-python>=3.4.5.20', 'torchnet', 'pandas', 'torchvision', 'visdom', 'matplotlib', 'imageio'
      ],
      zip_safe=False,
      python_requires='>=3.6.0'
      )
