from pydantic import BaseModel


class ForecastRequest(BaseModel):

    id: int

    store_nbr: int
    family: str
    onpromotion: int

    year: int
    month: int
    day: int
    day_of_week: int
    day_of_year: int
    week_of_year: int
    quarter: int

    is_weekend: int
    is_month_start: int
    is_month_end: int

    sales_lag_1: float
    sales_lag_7: float
    sales_lag_14: float
    sales_lag_28: float

    sales_rolling_mean_7: float
    sales_rolling_std_7: float

    sales_rolling_mean_14: float
    sales_rolling_std_14: float

    sales_rolling_mean_28: float
    sales_rolling_std_28: float

    city: str
    state: str
    type: str
    cluster: int

    transactions: float
    dcoilwtico: float

    holiday_count: int
    is_holiday: int


class ForecastResponse(BaseModel):

    predicted_sales: float