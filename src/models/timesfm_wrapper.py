import timesfm
import torch
import pandas as pd

class TimesFMWrapper:
    def __init__(self, context_len=2048, horizon_len=96, backend=None):
        self.device = backend if backend else ("gpu" if torch.cuda.is_available() else "cpu")
        self.tfm = timesfm.TimesFm(
            context_len=context_len,
            horizon_len=horizon_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            backend=self.device
        )
        self.checkpoint_repo = "google/timesfm-2.0-500m-pytorch"

    def load(self):
        print(f"Loading TimesFM from {self.checkpoint_repo}...")
        self.tfm.load_from_checkpoint(repo_id=self.checkpoint_repo)
        print("✅ Model loaded.")

    def forecast(self, df, freq="H", value_name="sales"):
        """
        Runs zero-shot inference on a dataframe.
        """
        forecast_df = self.tfm.forecast_on_df(
            inputs=df,
            freq=freq,
            value_name=value_name,
            num_jobs=-1
        )
        return forecast_df
