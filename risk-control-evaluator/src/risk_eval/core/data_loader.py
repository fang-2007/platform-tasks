"""Data loading with Strategy pattern for multiple formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from risk_eval.utils.exceptions import DataLoadError
from risk_eval.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loading strategies."""

    @abstractmethod
    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load data from source into a DataFrame.

        Args:
            source: Data source (file path or connection string).
            **kwargs: Additional loader-specific arguments.

        Returns:
            Loaded data as a pandas DataFrame.

        Raises:
            DataLoadError: If loading fails.
        """

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return set of supported file extensions."""


class CSVDataLoader(DataLoader):
    """Strategy for loading CSV files."""

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load CSV file.

        Args:
            source: Path to CSV file.
            **kwargs: Passed to pandas.read_csv.

        Returns:
            DataFrame with loaded data.
        """
        try:
            path = Path(source)
            logger.info("Loading CSV", path=str(path))
            df = pd.read_csv(path, **kwargs)
            logger.info("CSV loaded", rows=len(df), columns=len(df.columns))
            return df
        except Exception as exc:
            raise DataLoadError(
                f"Failed to load CSV: {exc}",
                context={"source": str(source)},
            ) from exc

    def supported_extensions(self) -> set[str]:
        return {".csv", ".txt"}


class ExcelDataLoader(DataLoader):
    """Strategy for loading Excel files."""

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load Excel file.

        Args:
            source: Path to Excel file.
            **kwargs: Passed to pandas.read_excel. Use 'sheet_name' to specify sheet.

        Returns:
            DataFrame with loaded data.
        """
        try:
            path = Path(source)
            sheet = kwargs.pop("sheet_name", 0)
            logger.info("Loading Excel", path=str(path), sheet=sheet)
            df = pd.read_excel(path, sheet_name=sheet, **kwargs)
            logger.info("Excel loaded", rows=len(df), columns=len(df.columns))
            return df
        except Exception as exc:
            raise DataLoadError(
                f"Failed to load Excel: {exc}",
                context={"source": str(source)},
            ) from exc

    def supported_extensions(self) -> set[str]:
        return {".xls", ".xlsx", ".xlsm"}


class JSONDataLoader(DataLoader):
    """Strategy for loading JSON files."""

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load JSON file.

        Args:
            source: Path to JSON file.
            **kwargs: Passed to pandas.read_json.

        Returns:
            DataFrame with loaded data.
        """
        try:
            path = Path(source)
            logger.info("Loading JSON", path=str(path))
            df = pd.read_json(path, **kwargs)
            logger.info("JSON loaded", rows=len(df), columns=len(df.columns))
            return df
        except Exception as exc:
            raise DataLoadError(
                f"Failed to load JSON: {exc}",
                context={"source": str(source)},
            ) from exc

    def supported_extensions(self) -> set[str]:
        return {".json", ".jsonl"}


class ParquetDataLoader(DataLoader):
    """Strategy for loading Parquet files."""

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load Parquet file.

        Args:
            source: Path to Parquet file.
            **kwargs: Passed to pandas.read_parquet.

        Returns:
            DataFrame with loaded data.
        """
        try:
            path = Path(source)
            logger.info("Loading Parquet", path=str(path))
            df = pd.read_parquet(path, **kwargs)
            logger.info("Parquet loaded", rows=len(df), columns=len(df.columns))
            return df
        except Exception as exc:
            raise DataLoadError(
                f"Failed to load Parquet: {exc}",
                context={"source": str(source)},
            ) from exc

    def supported_extensions(self) -> set[str]:
        return {".parquet"}


class SQLDataLoader(DataLoader):
    """Strategy for loading data from SQL databases."""

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load data from SQL query.

        Args:
            source: SQLAlchemy connection string.
            **kwargs: Must include 'query' key with SQL query string.

        Returns:
            DataFrame with query results.
        """
        query = kwargs.get("query")
        if not query:
            raise DataLoadError(
                "SQL loader requires 'query' parameter",
                context={"source": str(source)},
            )
        try:
            logger.info("Loading from SQL", source=str(source)[:50])
            engine = create_engine(str(source))
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            logger.info("SQL loaded", rows=len(df), columns=len(df.columns))
            return df
        except Exception as exc:
            raise DataLoadError(
                f"Failed to load SQL: {exc}",
                context={"source": str(source)[:50]},
            ) from exc

    def supported_extensions(self) -> set[str]:
        return set()  # SQL uses connection strings, not file extensions


class DataLoaderFactory:
    """Factory for creating appropriate DataLoader instances."""

    _loaders: list[type[DataLoader]] = [
        CSVDataLoader,
        ExcelDataLoader,
        JSONDataLoader,
        ParquetDataLoader,
        SQLDataLoader,
    ]

    @classmethod
    def create(cls, source: str | Path) -> DataLoader:
        """Create a DataLoader based on the source type.

        Args:
            source: File path or connection string.

        Returns:
            Appropriate DataLoader instance.

        Raises:
            DataLoadError: If no suitable loader is found.
        """
        source_str = str(source).lower()

        # SQL connection strings typically contain '://'
        if "://" in source_str:
            return SQLDataLoader()

        # File-based sources
        path = Path(source)
        suffix = path.suffix.lower()

        for loader_cls in cls._loaders:
            loader = loader_cls()
            if suffix in loader.supported_extensions():
                return loader

        supported = []
        for loader_cls in cls._loaders:
            supported.extend(loader_cls().supported_extensions())

        raise DataLoadError(
            f"Unsupported data source: {source}",
            context={
                "supported_extensions": ", ".join(sorted(set(supported))),
                "sql_pattern": "*://*",
            },
        )

    @classmethod
    def load(cls, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Convenience method: create loader and load data in one call.

        Args:
            source: Data source.
            **kwargs: Additional arguments for the loader.

        Returns:
            Loaded DataFrame.
        """
        loader = cls.create(source)
        return loader.load(source, **kwargs)
