import pytest

from trading_backtest.data import load_price_data, DataFormatError


def test_load_price_data_file_not_found(tmp_path):
    missing = tmp_path / "no_file.csv"
    with pytest.raises(FileNotFoundError):
        load_price_data(missing)


def test_load_price_data_bad_format(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text('a,b\n1,"2')
    with pytest.raises(DataFormatError):
        load_price_data(bad)
