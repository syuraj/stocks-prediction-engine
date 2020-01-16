# %%
import pytest
from stock_trainer import StockTrainer


# @pytest.mark.integration
# def test_create_prophet_model():
# %%
trainer = StockTrainer("", "TSLA")
model, future = trainer.create_prophet_model()

print(future)
# assert model is not None


# %%
