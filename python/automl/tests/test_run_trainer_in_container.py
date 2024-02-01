import os
import subprocess

class TestRunContainer:
    def test_run_resnet(self):
        script_path = os.path.abspath(os.path.join(os.pardir, 'autotrain', 'trainings', 'image_data', 'run.sh'))
        subprocess.run(['sh', script_path])