from datetime import date
from pathlib import Path

import pandas as pd

from app.core.config import settings
from config.clickhouse import get_client


SQL_DIR = Path(__file__).parent.parent / "sql"


def _read_sql(filename: str) -> str:
    return (SQL_DIR / filename).read_text(encoding="utf-8")


def _client_query(filters: str, *, include_target: bool) -> str:
    return _read_sql("feature_client.sql").format(
        filters=filters,
        target_column="dpd.MaxOverdueDays90," if include_target else "",
        target_join=(
            "LEFT JOIN risk_ch_db.dpd_495credit dpd "
            "ON cdc.ApplicationId = dpd.ApplicationId"
            if include_target else ""
        ),
    )


def _date_parameters(start_date: str | None, end_date: str | None) -> dict:
    start = date.fromisoformat(start_date or settings.TRAIN_START_DATE)
    end = date.fromisoformat(end_date or settings.TRAIN_END_DATE)
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")
    return {"start_date": start, "end_date": end}


def load_equifax_features(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    parameters = _date_parameters(start_date, end_date)
    query = _read_sql("feature_equifax.sql").format(
        filters="ec.ApplicationDate >= {start_date:Date} AND ec.ApplicationDate < {end_date:Date}",
    )
    return get_client().query_df(query, parameters=parameters)


def load_client_features(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    parameters = _date_parameters(start_date, end_date)
    query = _client_query(
        "cdc.ApplicationStatus IN ('LoanIssued', 'LoanReturned') "
        "AND cdc.ApplicationDate >= {start_date:Date} AND cdc.ApplicationDate < {end_date:Date}",
        include_target=True,
    )
    # Unmatched outcomes must remain NULL, not ClickHouse's default numeric zero.
    return get_client().query_df(query, parameters=parameters, settings={"join_use_nulls": 1})


def load_application_features(application_id: int) -> pd.DataFrame:
    """Use training feature expressions for one application, without its outcome."""
    if isinstance(application_id, bool) or not isinstance(application_id, int) or application_id < 1:
        raise ValueError("application_id must be a positive integer")

    client = get_client()
    parameters = {"application_id": application_id}
    clients = client.query_df(
        _client_query("cdc.ApplicationId = {application_id:UInt64}", include_target=False),
        parameters=parameters,
    )
    if clients.empty:
        return clients
    if len(clients) != 1:
        raise ValueError(f"Multiple client rows for ApplicationId {application_id}")

    equifax = client.query_df(
        _read_sql("feature_equifax.sql").format(
            filters="ec.ApplicationId = {application_id:UInt64}",
        ),
        parameters=parameters,
    )
    if equifax.empty:
        raise ValueError(f"Credit history features not found for ApplicationId {application_id}")
    if len(equifax) != 1:
        raise ValueError(f"Multiple credit history rows for ApplicationId {application_id}")

    dataset = clients.merge(
        equifax,
        how="inner",
        left_on="ApplicationId",
        right_on="ec.ApplicationId",
        validate="one_to_one",
    )
    if dataset.empty:
        raise ValueError(f"Credit history does not match ApplicationId {application_id}")
    return dataset
