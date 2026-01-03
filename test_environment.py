"""
Test script to validate the project environment setup.

This script imports key libraries (numpy, pandas, scikit-learn, mlflow)
to ensure they are installed and working correctly.
"""

import numpy as np
import pandas as pd
import sklearn
import mlflow

print("✅ Environment setup successful!")
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)
print("mlflow version:", mlflow.__version__)
