class FeatureEngineer:
    """Create features for portfolio optimization models."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize feature engineer."""
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
