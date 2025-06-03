from setuptools import setup

package_name = 'game'

setup(
    name=package_name,
    version='0.0.1',
    packages=['nodes', 'agents', 'configs', 'models'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wispytrace',
    maintainer_email='1025489007@qq.com',
    description='distributed Nash equilibrium seeking simulator',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node = nodes.node:main',
            'sync = nodes.sync:main'
        ],
    },
)
