import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

import radtree


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def iris_data():
    X, y = datasets.load_iris(return_X_y=True)
    return X, y


def test_plot_radial_smoke_with_ndarray_input(iris_data):
    X, y = iris_data
    X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_df, y)

    fig, ax = radtree.plot_radial(clf, X=X, Y=y, random_state=42)

    assert fig is not None
    assert ax is not None
    assert len(fig.axes) == 1


def test_plot_radial_num_samples_none_uses_full_dataset_without_sampling(
    monkeypatch, iris_data
):
    X, y = iris_data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=["target"])
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_df, y_df["target"])

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "DataFrame.sample should not be called when num_samples=None"
        )

    monkeypatch.setattr(pd.DataFrame, "sample", fail_if_called)

    fig, ax = radtree.plot_radial(
        clf, X=X_df, Y=y_df, num_samples=None, random_state=42
    )

    assert fig is not None
    assert ax is not None


def test_plot_radial_sampling_is_deterministic_for_same_random_state(
    monkeypatch, iris_data
):
    X, y = iris_data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=["target"])
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_df, y_df["target"])

    calls = []
    original_sample = pd.DataFrame.sample

    def recording_sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
        ignore_index=False,
    ):
        sampled = original_sample(
            self,
            n=n,
            frac=frac,
            replace=replace,
            weights=weights,
            random_state=random_state,
            axis=axis,
            ignore_index=ignore_index,
        )
        calls.append((n, random_state, tuple(sampled.index.tolist())))
        return sampled

    monkeypatch.setattr(pd.DataFrame, "sample", recording_sample)

    radtree.plot_radial(clf, X=X_df, Y=y_df, num_samples=25, random_state=7)
    radtree.plot_radial(clf, X=X_df, Y=y_df, num_samples=25, random_state=7)

    assert len(calls) >= 2
    assert calls[0][0] == 25
    assert calls[0][1] == 7
    assert calls[0][2] == calls[1][2]


def test_quick_fitted_tree_feature_selection_supports_pandas_inputs(iris_data):
    X, y = iris_data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    model, support_mask, splitted = radtree.quick_fitted_tree(
        X_df, y_series, model_type=["FeatureSelection"], random_state=42
    )

    assert model is not None
    assert support_mask is not None
    assert len(support_mask) == X_df.shape[1]
    assert isinstance(splitted, tuple)
