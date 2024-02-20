import os
import subprocess

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

class TestRunContainer:
    def test_run_resnet(self):
        script_path = os.path.join(PARENT_DIR, 'autotrain', 'trainings', 'image_data', 'run.sh')
        subprocess.run(['sh', script_path])