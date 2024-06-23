from setuptools import find_packages, setup

package_name = 'slam_rbpf'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/slam_rbpf_launch.py']),
        ('share/' + package_name + '/config', ['config/slam_rbpf.rviz']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='guangtian gong',
    maintainer_email='guangtian0330@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_rbpf_nav = slam_rbpf.slam_nav:main',
        ],
    },
)
