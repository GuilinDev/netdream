"""Locust workload generator for Online Boutique.

Run with:
    locust -f locustfile.py -H http://<frontend-host>:<port> \
           --headless -u <users> -r <spawn-rate> -t <duration>

Workload patterns are selected via LOCUST_WORKLOAD env var: constant | variable | bursty | flash.
We keep this file independent of NetDream so any baseline (HPA, PPO, random, GraphPilot)
can reuse it identically.
"""

import math
import os
import random
import time

from locust import HttpUser, LoadTestShape, between, events, task


# ---------------------------------------------------------------------------
# User behavior — same pattern Online Boutique's upstream loadgenerator uses
# ---------------------------------------------------------------------------

PRODUCTS = [
    "0PUK6V6EV0", "1YMWWN1N4O", "2ZYFJ3GM2N", "66VCHSJNUP", "6E92ZMYYFZ",
    "9SIQT8TOJO", "L9ECAV7KIM", "LS4PSXUNUM", "OLJCESPC7Z",
]


class BoutiqueUser(HttpUser):
    wait_time = between(1.0, 3.0)

    @task(10)
    def index(self):
        self.client.get("/", name="index")

    @task(5)
    def browse_product(self):
        pid = random.choice(PRODUCTS)
        self.client.get(f"/product/{pid}", name="product")

    @task(3)
    def set_currency(self):
        self.client.post("/setCurrency", data={"currency_code": random.choice(["USD", "EUR", "JPY"])},
                         name="setCurrency")

    @task(2)
    def add_to_cart(self):
        pid = random.choice(PRODUCTS)
        self.client.post("/cart",
                         data={"product_id": pid, "quantity": random.choice([1, 2, 3])},
                         name="cart")

    @task(1)
    def view_cart(self):
        self.client.get("/cart", name="cartview")


# ---------------------------------------------------------------------------
# Workload shapes
# ---------------------------------------------------------------------------

class ConstantShape(LoadTestShape):
    """50 concurrent users, steady for the whole run."""
    use_common_options = True
    target = 50
    duration = 60 * 60  # 1 hour
    def tick(self):
        t = self.get_run_time()
        if t >= self.duration:
            return None
        return (self.target, 10)


class VariableShape(LoadTestShape):
    """Sine wave between 20 and 80 users, 180 s period."""
    duration = 60 * 60
    def tick(self):
        t = self.get_run_time()
        if t >= self.duration:
            return None
        users = int(50 + 30 * math.sin(2 * math.pi * t / 180.0))
        return (max(1, users), 20)


class BurstyShape(LoadTestShape):
    """Base load 30 users; random 15 s burst to 120 users every ~90 s."""
    duration = 60 * 60
    _burst_end = 0
    def tick(self):
        t = self.get_run_time()
        if t >= self.duration:
            return None
        if t > self._burst_end and random.random() < 1 / 90.0:
            self._burst_end = t + 15
        if t <= self._burst_end:
            return (120, 50)
        return (30, 10)


class FlashCrowdShape(LoadTestShape):
    """Ramp from 10 → 200 users over 60 s, sustained for 5 min, then back to 10."""
    duration = 60 * 60
    def tick(self):
        t = self.get_run_time()
        if t >= self.duration:
            return None
        if t < 60:
            return (int(10 + (190 * t / 60)), 50)
        if t < 360:
            return (200, 50)
        if t < 420:
            return (int(200 - 190 * (t - 360) / 60), 50)
        return (10, 10)


# Shape selection via env var
_SHAPES = {
    "constant": ConstantShape,
    "variable": VariableShape,
    "bursty":   BurstyShape,
    "flash":    FlashCrowdShape,
}


@events.init.add_listener
def _on_locust_init(environment, **kwargs):
    mode = os.environ.get("LOCUST_WORKLOAD", "constant")
    shape_cls = _SHAPES.get(mode)
    if shape_cls is None:
        raise RuntimeError(f"Unknown LOCUST_WORKLOAD={mode!r}; choose from {list(_SHAPES)}")
    environment.shape_class = shape_cls()
    print(f"[locust] workload={mode} shape={shape_cls.__name__}")
