import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regimes (Bull, Bear, Sideways) using Hidden Markov Models.
    Based on log returns and volatility.
    """

    def __init__(self, n_components: int = 3, n_iter: int = 100):
        """
        Initialize the RegimeDetector.

        Args:
            n_components: Number of hidden states (default: 3 for Bull, Bear, Sideways)
            n_iter: Max iterations for HMM training
        """
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            n_iter=n_iter,
            random_state=42
        )
        self.state_map = {}  # Map hidden state index to label

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features returning a DataFrame with NaNs dropped.
        """
        data = df.copy()
        data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))
        data["volatility"] = data["log_ret"].rolling(window=20).std()
        return data.dropna()

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM training.
        """
        data = self._compute_features(df)
        if data.empty:
             raise ValueError("Not enough data to compute features (need > 20 days)")
        return np.column_stack([data["log_ret"], data["volatility"]])

    def fit(self, df: pd.DataFrame):
        """
        Train the HMM on historical data.
        """
        try:
            X = self.prepare_features(df)
            self.model.fit(X)
            
            # Map states to labels based on mean returns: 0=Bear, 2=Bull
            means = self.model.means_[:, 0]
            sorted_indices = np.argsort(means)
            
            self.state_map = {
                sorted_indices[0]: "Bear",
                sorted_indices[-1]: "Bull"
            }
            if self.n_components == 3:
                self.state_map[sorted_indices[1]] = "Sideways"
            
            logger.info(f"Regime model fitted. State map: {self.state_map}")

        except Exception as e:
            logger.error(f"Failed to fit regime model: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regimes for the input DataFrame.
        """
        if not self.state_map:
             raise RuntimeError("Model not fitted. Call fit() first.")

        try:
            valid_data = self._compute_features(df)
            if valid_data.empty:
                return df

            X = np.column_stack([valid_data["log_ret"], valid_data["volatility"]])
            hidden_states = self.model.predict(X)
            
            # Create a localized series for alignment
            regime_series = pd.Series(hidden_states, index=valid_data.index, name="Regime_State")
            
            # Join back to original DF
            result_df = df.copy()
            result_df["Regime_State"] = regime_series
            result_df["Regime"] = result_df["Regime_State"].map(self.state_map)
            
            return result_df

        except Exception as e:
            logger.error(f"Failed to predict regimes: {e}")
            return df
