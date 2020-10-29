from setuptools import setup

setup(name='pytorch-neural-cellular-automata',
      version='0.1',
      description='An pytorch implementation of Growing Neural Cellular Automata',
      url='https://github.com/SethPipho/pytorch-neural-cellular-automata',
      author='Seth Pipho',
      author_email='seth.pipho@gmail.com',
      license='MIT',
      packages=['pytorch_neural_ca'],
       install_requires=[
          'torch',
          'torchvision',
          'pillow',
          'click'
      ],
      entry_points = {
        'console_scripts': ['pytorch_neural_ca=pytorch_neural_ca.__main__:main'],
      },
      zip_safe=True)