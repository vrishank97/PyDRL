import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PyDRL',
    version='0.0.1',
    description='A Python Deep Reinforcement Learning library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    author='Vrishank Bhardwaj',
    author_email='vrishank1997@gmail.com',
    url='https://github.com/vrishank97/PyDRL',
    packages=setuptools.find_packages()
)