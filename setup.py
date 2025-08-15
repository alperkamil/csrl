from setuptools import setup

setup(
      name='csrl',
      version='2.0',
      description='Control Synthesis from Formal Specifications using Reinforcement Learning',
      url='https://github.com/alperkamil/csrl',
      author='Alper Kamil Bozkurt',
      license='MIT License',
      packages=['csrl'],
      zip_safe=False,
      install_requires=[
            'numpy>=2',
            'gymnasium>=1',
            'matplotlib>=3.8',
            'ipywidgets>=8',
            'spot>=2.11',
            'numba>=0.57',
      ]
)