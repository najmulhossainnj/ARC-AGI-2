from __future__ import annotations
import os
import json
import time
import argparse
import requests
from pathlib import Path

# Config defaults
DEFAULT_SERVER_URL = os.environ.get("ARC_COORDINATOR_URL", "http://localhost:5000")


class CloudWorkerClient:
    def __init__(self, server_url: str, worker_id: str):
        self.server_url = server_url.rstrip("/")
        self.worker_id = worker_id

    def get_task(self):
        """Fetch next available task from central coordinator."""
        try:
            resp = requests.post(f"{self.server_url}/api/get_task", json={"worker_id": self.worker_id}, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error requesting task: {e}")
        return None

    def submit_result(self, task_id: str, solved: bool, source: str, program_str: str, new_primitive_code: str = None):
        """Report task result and sync any discovered new primitive."""
        payload = {
            "worker_id": self.worker_id,
            "task_id": task_id,
            "solved": solved,
            "source": source,
            "program_str": program_str,
            "new_primitive_code": new_primitive_code,
        }
        try:
            resp = requests.post(f"{self.server_url}/api/submit_result", json=payload, timeout=15)
            return resp.status_code == 200
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error submitting result for {task_id}: {e}")
            return False

    def sync_primitives(self) -> list:
        """Fetch global primitives list from master coordinator."""
        try:
            resp = requests.get(f"{self.server_url}/api/primitives", timeout=10)
            if resp.status_code == 200:
                return resp.json().get("primitives", [])
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error syncing primitives: {e}")
        return []
