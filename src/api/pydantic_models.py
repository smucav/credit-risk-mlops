from pydantic import BaseModel
from typing import Optional


class CustomerInput(BaseModel):
    Amount: float
    Value: float
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    PricingStrategy: int
    FraudResult: int
    Value_mean: Optional[float] = 9900.0
    Recency: Optional[int] = 0  # FIXED: Days since last txn
    Frequency: Optional[int] = 50  # FIXED: Txn count
    ChannelId: str
    ProductCategory: str
    ProviderId: str
    ProductId: str
    CurrencyCode: Optional[str] = "UGX"
    CountryCode: Optional[int] = 256


class RiskPrediction(BaseModel):
    risk_probability: float
    is_high_risk: bool
    credit_score: float
