"""Thin Prometheus HTTP client for cluster-side state collection.

Only the instant-query surface we need for the NetDream K8s controller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class PromClient:
    url: str  # e.g. "http://localhost:9090"
    timeout: float = 5.0

    def query(self, promql: str) -> List[dict]:
        endpoint = f"{self.url}/api/v1/query?{urlencode({'query': promql})}"
        with urlopen(endpoint, timeout=self.timeout) as r:
            body = r.read().decode()
        import json
        data = json.loads(body)
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data["data"]["result"]

    def scalar_by_label(self, promql: str, label: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for row in self.query(promql):
            key = row["metric"].get(label)
            if key is None:
                continue
            out[key] = float(row["value"][1])
        return out


def wait_ready(client: PromClient, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            client.query("up")
            return
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Prometheus not ready after {timeout_s}s: {last_err}")
