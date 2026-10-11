"""Tests for data loading strategies."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from risk_eval.core.data_loader import (
    CSVDataLoader,
    DataLoaderFactory,
    ExcelDataLoader,
    JSONDataLoader,
    ParquetDataLoader,
    SQLDataLoader,
)
from risk_eval.utils.exceptions import DataLoadError


class TestCSVDataLoader:
    """Tests for CSV loading strategy."""

    def test_load_valid_csv(self, temp_csv_file: Path, sample_dataframe: pd.DataFrame) -> None:
        """Test loading a valid CSV file."""
        loader = CSVDataLoader()
        result = loader.load(temp_csv_file)

        assert len(result) == len(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)

    def test_supported_extensions(self) -> None:
        """Test that CSV extensions are reported."""
        loader = CSVDataLoader()
        assert loader.supported_extensions() == {".csv", ".txt"}

    def test_load_nonexistent_file(self) -> None:
        """Test loading a non-existent file raises DataLoadError."""
        loader = CSVDataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load("/nonexistent/file.csv")
        assert "Failed to load CSV" in str(exc_info.value)


class TestExcelDataLoader:
    """Tests for Excel loading strategy."""

    def test_load_valid_excel(self, temp_excel_file: Path, sample_dataframe: pd.DataFrame) -> None:
        """Test loading a valid Excel file."""
        loader = ExcelDataLoader()
        result = loader.load(temp_excel_file)

        assert len(result) == len(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)

    def test_supported_extensions(self) -> None:
        """Test that Excel extensions are reported."""
        loader = ExcelDataLoader()
        assert loader.supported_extensions() == {".xls", ".xlsx", ".xlsm"}


class TestJSONDataLoader:
    """Tests for JSON loading strategy."""

    def test_load_valid_json(self, sample_dataframe: pd.DataFrame) -> None:
        """Test loading a valid JSON file."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)
        sample_dataframe.to_json(path, orient="records")

        loader = JSONDataLoader()
        result = loader.load(path)

        assert len(result) == len(sample_dataframe)
        path.unlink(missing_ok=True)


class TestParquetDataLoader:
    """Tests for Parquet loading strategy."""

    def test_load_valid_parquet(self, sample_dataframe: pd.DataFrame) -> None:
        """Test loading a valid Parquet file."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        sample_dataframe.to_parquet(path, index=False)

        loader = ParquetDataLoader()
        result = loader.load(path)

        assert len(result) == len(sample_dataframe)
        path.unlink(missing_ok=True)


class TestSQLDataLoader:
    """Tests for SQL loading strategy."""

    def test_supported_extensions_empty(self) -> None:
        """Test that SQL loader has no file extensions."""
        loader = SQLDataLoader()
        assert loader.supported_extensions() == set()

    def test_load_without_query_raises(self) -> None:
        """Test that loading without query parameter raises error."""
        loader = SQLDataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load("sqlite:///:memory:")
        assert "requires 'query' parameter" in str(exc_info.value)


class TestDataLoaderFactory:
    """Tests for DataLoaderFactory."""

    @pytest.mark.parametrize("source,expected_class", [
        ("data.csv", CSVDataLoader),
        ("data.xlsx", ExcelDataLoader),
        ("data.json", JSONDataLoader),
        ("data.parquet", ParquetDataLoader),
        ("sqlite:///:memory:", SQLDataLoader),
    ])
    def test_create_by_extension(self, source: str, expected_class: type) -> None:
        """Test factory creates correct loader by file extension."""
        loader = DataLoaderFactory.create(source)
        assert isinstance(loader, expected_class)

    def test_create_unsupported_extension_raises(self) -> None:
        """Test factory raises error for unsupported extension."""
        with pytest.raises(DataLoadError) as exc_info:
            DataLoaderFactory.create("data.unknown")
        assert "Unsupported data source" in str(exc_info.value)

    def test_load_convenience_method(self, temp_csv_file: Path, sample_dataframe: pd.DataFrame) -> None:
        """Test factory load convenience method."""
        result = DataLoaderFactory.load(temp_csv_file)
        assert len(result) == len(sample_dataframe)
