from glob import glob
from setuptools import setup

setup(
    name='hercules_maicp', version='0.1.0', packages=['hercules_maicp'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/hercules_maicp']),
                ('share/hercules_maicp', ['package.xml', 'README.md']),
                ('share/hercules_maicp/launch', glob('launch/*.launch.py'))],
    install_requires=['setuptools', 'numpy', 'cvxpy>=1.4,<2'],
    entry_points={'console_scripts': [
        'calibration_worker = hercules_maicp.ros_transport:worker_main',
        'experiment = hercules_maicp.experiment:main',
        'reset_scene = hercules_maicp.reset_scene:main',
        'fit_dynamics = hercules_maicp.dynamics:_cli',
    ]},
)
