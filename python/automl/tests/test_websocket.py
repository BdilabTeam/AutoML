import pytest
from alserver.rest.app import create_app
from alserver.handlers.dataplane import DataPlane
from alserver import Settings
from fastapi.testclient import TestClient
from fastapi import FastAPI

EXPERIMENT_JOB_NAME = ""

class TestWebSocket:
    @pytest.fixture(scope='class')
    def app(self):
        settings = Settings(
            kubernetes_enabled=True,
            http_port=31185
        )
        dataplane = DataPlane(settings=settings)
        return create_app(data_plane=dataplane)
    
    def test_websocket(self, app: FastAPI):
        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/experiment/job/logs?experiment_job_name={EXPERIMENT_JOB_NAME}") as websocket:
            try:
                while True:
                    data = websocket.receive()
                    print(data)
            except Exception as e:
                print("WebSocket closed:", e)
            finally:
                websocket.close()             