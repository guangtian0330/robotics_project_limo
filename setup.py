from setuptools import find_packages, setup
import os

package_name = 'slam_rbpf'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), [os.path.join('launch', 'gazebo_models_diff.launch.py')]),
        (os.path.join('share', package_name, 'launch'), [os.path.join('launch', 'slam_rbpf_gazebo.launch.py')]),
        (os.path.join('share', package_name, 'launch'), [os.path.join('launch', 'slam_rbpf_limo.launch.py')]),
        (os.path.join('share', package_name, 'meshes'), [os.path.join('meshes', 'limo_base.dae')]), 
        (os.path.join('share', package_name, 'meshes'), [os.path.join('meshes', 'limo_base.stl')]), 
        (os.path.join('share', package_name, 'meshes'), [os.path.join('meshes', 'limo_wheel.dae')]), 
        (os.path.join('share', package_name, 'meshes'), [os.path.join('meshes', 'limo_wheel.stl')]), 
        (os.path.join('share', package_name, 'rviz'), [os.path.join('rviz', 'model_display.rviz')]),
        (os.path.join('share', package_name, 'rviz'), [os.path.join('rviz', 'slam_rbpf.rviz')]),
        (os.path.join('share', package_name, 'rviz'), [os.path.join('rviz', 'urdf.rviz')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_four_diff.xacro')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_ackerman.xacro')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_steering_hinge.xacro')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_xacro.xacro')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_ackerman.gazebo')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_four_diff_2.gazebo')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_four_diff.gazebo')]),
        (os.path.join('share', package_name, 'urdf'), [os.path.join('urdf', 'limo_gazebo.gazebo')]),
        (os.path.join('share', package_name, 'worlds'), [os.path.join('worlds', 'basic.world')]),
        (os.path.join('share', package_name, 'worlds'), [os.path.join('worlds', 'simple_world.world')]),
        (os.path.join('share', package_name, 'worlds'), [os.path.join('worlds', 'simple_hallway2.world')]),
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
            'map_display = slam_rbpf.map_display:main',
            'path_node = slam_rbpf.path_planner:main'
        ],
    },
)
