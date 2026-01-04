import sys
from pathlib import Path

import lightning as L
from pytorch_forecasting import DeepAR

print(f"⚡ Lightning Version: {L.__version__}")

# Create a dummy DeepAR (just the class, no data needed for isinstance check)
# DeepAR inherits from BaseModel -> LightningModule
# We don't need to initialize it fully to check the class inheritance structure
print(f"🔍 Checking inheritance...")

if issubclass(DeepAR, L.LightningModule):
    print("✅ SUCCESS: DeepAR is a subclass of LightningModule.")
else:
    print("❌ FAILURE: DeepAR is NOT a subclass of LightningModule.")


# Double check with an instance (requires minimal init args)
try:
    # We can't easily instantiate DeepAR without a dataset,
    # but the class check above is usually sufficient for library compatibility.
    pass
except Exception as e:
    print(e)
