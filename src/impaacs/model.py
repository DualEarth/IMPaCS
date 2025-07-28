# src/impaacs/model.py

class IMPAaCS:
    def __init__(self, config: IMPAaCSConfig):
        # load grid, state, etc.
    def update(self, loc, diameter): …
    def run_sequence(self, impacts: List[ImpactEvent]) -> SimulationResult:
        # high‐level driver
