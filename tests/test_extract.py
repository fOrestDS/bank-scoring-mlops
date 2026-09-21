from datetime import date
from unittest.mock import Mock

import pandas as pd
import pytest

from ml import extract
from ml.dataset_builder import DatasetBuilder
from ml.preprocess import preprocess_dataset


@pytest.fixture
def client(monkeypatch):
    client = Mock()
    monkeypatch.setattr(extract, "get_client", lambda: client)
    return client


def client_rows():
    return pd.DataFrame({"ApplicationId": [123], "MainDebt": [10000], "OS": ["Android"]})


def history_rows():
    return pd.DataFrame({"ec.ApplicationId": [123], "active_count": [3], "active_debt": [45000]})


def test_by_id_matches_training_features(client):
    clients, history = client_rows(), history_rows()
    client.query_df.side_effect = [clients, history]
    actual = preprocess_dataset(extract.load_application_features(123))
    expected = DatasetBuilder().build_from_dataframes(
        clients.assign(MaxOverdueDays90=30), history,
    ).drop(columns="Target")
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[0, "active_debt"] == 45000

    calls = client.query_df.call_args_list
    client_sql, history_sql = [call.args[0] for call in calls]
    assert "dpd" not in client_sql
    assert "ApplicationStatus" not in client_sql
    assert "{application_id:UInt64}" in client_sql
    assert "GROUP BY ec.ApplicationId" in history_sql
    assert "as active_debt" in history_sql
    assert "ec.*" not in history_sql
    assert all(call.kwargs["parameters"] == {"application_id": 123} for call in calls)


def test_training_and_by_id_use_same_equifax_expressions(client):
    client.query_df.return_value = history_rows()
    extract.load_equifax_features("2025-06-01", "2025-08-01")
    training_sql = client.query_df.call_args.args[0]
    client.query_df.side_effect = [client_rows(), history_rows()]
    extract.load_application_features(123)
    scoring_sql = client.query_df.call_args.args[0]
    assert training_sql.rsplit("WHERE", 1)[0] == scoring_sql.rsplit("WHERE", 1)[0]


def test_unknown_application_returns_empty(client):
    client.query_df.return_value = client_rows().iloc[:0]
    assert extract.load_application_features(123).empty
    client.query_df.assert_called_once()


@pytest.mark.parametrize("case", ["duplicate_client", "duplicate_history", "no_history", "wrong_id"])
def test_invalid_feature_rows_are_rejected(client, case):
    clients, history = client_rows(), history_rows()
    if case == "duplicate_client":
        clients = pd.concat([clients, clients])
    elif case == "duplicate_history":
        history = pd.concat([history, history])
    elif case == "no_history":
        history = history.iloc[:0]
    else:
        history["ec.ApplicationId"] = 456
    client.query_df.side_effect = [clients, history]
    with pytest.raises(ValueError):
        extract.load_application_features(123)


@pytest.mark.parametrize("application_id", [0, -1, True, "123 OR 1=1"])
def test_invalid_application_id_does_not_query(client, application_id):
    with pytest.raises(ValueError):
        extract.load_application_features(application_id)
    client.query_df.assert_not_called()


@pytest.mark.parametrize("loader", [extract.load_client_features, extract.load_equifax_features])
def test_batch_loaders_use_default_dates(client, monkeypatch, loader):
    monkeypatch.setattr(extract.settings, "TRAIN_START_DATE", "2025-06-01")
    monkeypatch.setattr(extract.settings, "TRAIN_END_DATE", "2025-08-01")
    loader()
    assert client.query_df.call_args.kwargs["parameters"] == {
        "start_date": date(2025, 6, 1), "end_date": date(2025, 8, 1),
    }


def test_invalid_date_range_does_not_query(client):
    with pytest.raises(ValueError):
        extract.load_client_features("2025-08-01", "2025-06-01")
    client.query_df.assert_not_called()
