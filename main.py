"""
DLMDSPWP01 – Programming with Python
Single-file solution that:
1) Loads train/ideal/test CSVs
2) Creates SQLite DB tables via SQLAlchemy
3) Selects 4 ideal functions (Least Squares)
4) Maps test points to those ideals using √2 rule
5) Visualizes data with Bokeh (HTML, offline inline)
6) Uses OOP with inheritance + custom exceptions
7) Includes unit tests (run with: python assignment.py --test)
"""

# ===============================
# Imports
# ===============================
import math
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Bokeh for visualization
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool
from bokeh.palettes import Category10


# ===============================
# Custom Exceptions (Req: exceptions)
# ===============================
class AssignmentError(Exception):
    """Base exception for assignment errors."""


class DataAlignmentError(AssignmentError):
    """Raised when x-values between tables are not aligned."""


class MissingTableError(AssignmentError):
    """Raised when an expected table is missing in the database."""


# ===============================
# Data Sources (Req: OOP + inheritance)
# ===============================
class DataSource:
    """Abstract base class for data providers."""
    def load(self) -> pd.DataFrame:
        raise NotImplementedError


class CSVSource(DataSource):
    """CSV implementation of DataSource with optional column filtering."""
    def __init__(self, path: str, usecols: list[str] | None = None):
        self.path = path
        self.usecols = usecols

    def load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path)
            if self.usecols:
                df = df[self.usecols]
            return df
        except FileNotFoundError as e:
            raise AssignmentError(f"CSV not found: {self.path}") from e
        except Exception as e:
            raise AssignmentError(f"Failed to read CSV: {self.path}") from e


# ===============================
# Persistence Layer (SQLite via SQLAlchemy)
# ===============================
class SQLiteStore:
    def __init__(self, db_path: str = "assignment.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)

    def write(self, name: str, df: pd.DataFrame, replace: bool = True) -> None:
        if df is None or (df.empty and df.columns.empty):
            raise AssignmentError(f"Refusing to write empty/columnless table: {name}")
        df.to_sql(name, con=self.engine, if_exists=("replace" if replace else "fail"), index=False)

    def read(self, name: str) -> pd.DataFrame:
        try:
            with self.engine.begin() as conn:
                return pd.read_sql_table(name, conn)
        except ValueError as e:
            raise MissingTableError(f"Table not found: {name}") from e


# ===============================
# Analysis – Step 3: Select 4 best ideal functions (Least Squares)
# ===============================
class IdealSelector:
    """
    Selects, for each training series y1..y4, the ideal function (from 50) minimizing SSE.
    Also computes the max absolute deviation (used for √2 rule).
    """
    def __init__(self, store: SQLiteStore):
        self.store = store

    @staticmethod
    def _check_alignment(trn: pd.DataFrame, ideal: pd.DataFrame) -> None:
        if not np.allclose(trn["x"].values, ideal["x"].values):
            raise DataAlignmentError("Training and ideal x-values are not aligned.")

    def run(self) -> pd.DataFrame:
        trn = self.store.read("training_data")
        ideal = self.store.read("ideal_functions")
        self._check_alignment(trn, ideal)

        train_cols = ["y1", "y2", "y3", "y4"]
        ideal_cols = [c for c in ideal.columns if c.startswith("y")]

        rows = []
        for tcol in train_cols:
            y_train = trn[tcol].values
            # Compute SSE for each ideal function column
            best_col, best_sse = min(
                ((icol, float(np.sum((y_train - ideal[icol].values) ** 2))) for icol in ideal_cols),
                key=lambda t: t[1]
            )
            max_abs_dev = float(np.max(np.abs(y_train - ideal[best_col].values)))
            rows.append({
                "train_func": tcol,
                "ideal_func": best_col,
                "sse": best_sse,
                "max_abs_dev": max_abs_dev,
            })

        out = pd.DataFrame(rows, columns=["train_func", "ideal_func", "sse", "max_abs_dev"])
        self.store.write("selected_ideals", out)
        return out


# ===============================
# Analysis – Step 4: Map test data with √2 criterion
# ===============================
class TestMapper:
    """
    Maps each (x,y) in test data to one of the selected ideal functions if:
    |y_test - y_ideal(x)| <= max_training_deviation * sqrt(2)
    """
    def __init__(self, store: SQLiteStore):
        self.store = store

    def run(self, test_csv_path: str) -> pd.DataFrame:
        test_df = CSVSource(test_csv_path).load()
        selected = self.store.read("selected_ideals")
        ideal_df = self.store.read("ideal_functions")

        results = []
        for _, row in test_df.iterrows():
            x_val = row["x"]
            y_val = row["y"]

            # Try each selected ideal until a match is found (or none)
            for _, sel in selected.iterrows():
                ideal_col = sel["ideal_func"]
                max_dev_allowed = float(sel["max_abs_dev"]) * math.sqrt(2)

                # Find y_ideal at same x
                match = ideal_df[ideal_df["x"] == x_val]
                if match.empty:
                    continue
                y_ideal = float(match[ideal_col].values[0])
                deviation = abs(y_val - y_ideal)

                if deviation <= max_dev_allowed:
                    results.append({
                        "x_test": x_val,
                        "y_test": y_val,
                        "delta_y": deviation,
                        "ideal_func_no": ideal_col
                    })

        out = pd.DataFrame(results, columns=["x_test", "y_test", "delta_y", "ideal_func_no"])
        self.store.write("test_results", out)
        return out


# ===============================
# Visualization – Step 5: Bokeh HTML (inline, offline)
# ===============================
class Visualizer:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def save_html(self, filename: str = "visualization.html") -> None:
        trn = self.store.read("training_data")
        ideal = self.store.read("ideal_functions")
        selected = self.store.read("selected_ideals")
        testres = self.store.read("test_results")

        output_file(filename=filename, title="DLMDSPWP01 – Data & Mapping", mode="inline")

        p = figure(
            title="Training vs. Selected Ideals with Test Mapping",
            width=1100, height=650,
            x_axis_label="x", y_axis_label="y",
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            active_scroll="wheel_zoom"
        )
        p.add_tools(HoverTool(tooltips=[("x", "@x"), ("y", "@y")]))

        # Training series (dashed)
        for col in ["y1", "y2", "y3", "y4"]:
            p.line(trn["x"], trn[col], line_dash="dashed", line_width=1.5,
                   alpha=0.6, legend_label=f"train {col}")

        # Selected ideal functions (solid)
        palette = Category10[10]
        for i, row in selected.iterrows():
            ideal_col = row["ideal_func"]
            color = palette[i % len(palette)]
            p.line(ideal["x"], ideal[ideal_col], line_width=3, color=color,
                   legend_label=f"ideal {ideal_col}")

            # Mapped test points for this ideal
            ti = testres[testres["ideal_func_no"] == ideal_col].copy()
            if ti.empty:
                continue

            ideal_slice = ideal[["x", ideal_col]].rename(columns={ideal_col: "y_ideal"})
            ti = ti.merge(ideal_slice, left_on="x_test", right_on="x", how="left")

            p.scatter(ti["x_test"], ti["y_test"], size=6, color=color, alpha=0.9,
                      legend_label=f"mapped→ {ideal_col}")
            # deviation segments
            p.segment(x0=ti["x_test"], y0=ti["y_test"], x1=ti["x_test"], y1=ti["y_ideal"],
                      line_color=color, line_alpha=0.35)

        p.legend.click_policy = "hide"
        p.legend.location = "top_left"
        save(p)


# ===============================
# Orchestration – Steps 1 & 2 + whole pipeline
# ===============================
def run_pipeline(train_csv="train.csv", ideal_csv="ideal.csv", test_csv="test.csv",
                 db_path="assignment.db", out_html="visualization.html") -> None:
    """
    Steps 1–2: read CSVs, create DB & tables.
    Step 3: select ideals.
    Step 4: map tests.
    Step 5: visualize.
    """
    # Step 1: load CSVs
    train = CSVSource(train_csv, usecols=["x", "y1", "y2", "y3", "y4"]).load()
    ideal = CSVSource(ideal_csv).load()

    # Step 2: write tables (training, ideal, empty test_results)
    store = SQLiteStore(db_path)
    store.write("training_data", train)
    store.write("ideal_functions", ideal)
    store.write("test_results", pd.DataFrame(columns=["x_test", "y_test", "delta_y", "ideal_func_no"]))

    # Step 3: select ideals
    selected = IdealSelector(store).run()
    print("Selected ideals:\n", selected.to_string(index=False))

    # Step 4: map tests
    mapped = TestMapper(store).run(test_csv)
    print(f"Mapped test points: {len(mapped)}")

    # Step 5: visualize
    Visualizer(store).save_html(out_html)
    print(f"Saved visualization to: {out_html}")


# ===============================
# Unit Tests (Step 7) – run with: python assignment.py --test
# ===============================
def run_tests():
    import os
    import unittest

    class TestAssignment(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.db = "test_assignment.db"
            if os.path.exists(cls.db):
                os.remove(cls.db)
            cls.store = SQLiteStore(cls.db)

            # Load CSVs
            train = CSVSource("train.csv", usecols=["x", "y1", "y2", "y3", "y4"]).load()
            ideal = CSVSource("ideal.csv").load()

            cls.store.write("training_data", train)
            cls.store.write("ideal_functions", ideal)
            cls.store.write("test_results", pd.DataFrame(columns=["x_test", "y_test", "delta_y", "ideal_func_no"]))

        def test_selects_four(self):
            sel = IdealSelector(self.store).run()
            self.assertEqual(len(sel), 4)
            self.assertTrue({"train_func","ideal_func","sse","max_abs_dev"}.issubset(sel.columns))

        def test_mapping_respects_rule(self):
            # Prepare
            IdealSelector(self.store).run()
            mapped = TestMapper(self.store).run("test.csv")
            selected = self.store.read("selected_ideals")
            ideal = self.store.read("ideal_functions")

            # Verify √2 rule
            for _, r in mapped.iterrows():
                x_val, delta = r["x_test"], r["delta_y"]
                icol = r["ideal_func_no"]
                row = selected[selected["ideal_func"] == icol].iloc[0]
                max_allowed = float(row["max_abs_dev"]) * math.sqrt(2)
                self.assertLessEqual(delta, max_allowed + 1e-9)

        def test_alignment_guard(self):
            # corrupt alignment
            ideal = self.store.read("ideal_functions").copy()
            ideal.loc[0, "x"] += 0.12345
            self.store.write("ideal_functions", ideal)
            with self.assertRaises(DataAlignmentError):
                IdealSelector(self.store).run()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAssignment)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


# ===============================
# CLI
# ===============================
def parse_args():
    p = argparse.ArgumentParser(description="DLMDSPWP01 assignment pipeline")
    p.add_argument("--train", default="train.csv", help="Path to training CSV")
    p.add_argument("--ideal", default="ideal.csv", help="Path to ideal CSV")
    p.add_argument("--test",  default="test.csv",  help="Path to test CSV")
    p.add_argument("--db",    default="assignment.db", help="SQLite DB file")
    p.add_argument("--html",  default="visualization.html", help="Output HTML for plot")
    p.add_argument("--test-suite", dest="run_test_suite", action="store_true",
                   help="Run built-in unit tests instead of the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    if args.run_test_suite:
        run_tests()
    else:
        run_pipeline(train_csv=args.train, ideal_csv=args.ideal, test_csv=args.test,
                     db_path=args.db, out_html=args.html)


if __name__ == "__main__":
    main()
