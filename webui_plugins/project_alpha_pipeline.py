from __future__ import annotations
import os
import subprocess
import pandas as pd
from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint

class Pipeline(FunctionCallingBlueprint):
    """Open WebUI pipeline to run Project Alpha tasks."""

    class Valves(FunctionCallingBlueprint.Valves):
        MARKET: str = "us"
        pass

    class Tools:
        def __init__(self, pipeline: 'Pipeline') -> None:
            self.pipeline = pipeline

        def run_full_scan(self, market: str | None = None) -> str:
            """Run the stock scanner for the given market."""
            market = market or self.pipeline.valves.MARKET
            cmd = ["python", "src/project_alpha.py", "--market", market]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout

        def get_latest_trending(self, limit: int = 10) -> list[str]:
            """Return the latest trending symbols."""
            market = self.pipeline.valves.MARKET
            fname = f"data/historic_results/{market}/screener_trend.csv"
            if not os.path.exists(fname):
                return []
            df = pd.read_csv(fname, header=None)
            if df.empty:
                return []
            last_row = df.tail(1).iloc[0, 1:].dropna()
            return last_row.tolist()[:limit]

    def __init__(self) -> None:
        super().__init__()
        self.name = "Project Alpha Pipeline"
        self.valves = self.Valves(**{**self.valves.model_dump()})
        self.tools = self.Tools(self)
