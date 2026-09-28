import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    import numpy as np
    from catboost import CatBoostClassifier
    import pandas as pd

    from metric import precision_at_recall, pr_curve

    pl.Config.set_tbl_rows(-1)
    return CatBoostClassifier, np, pl, precision_at_recall


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Изначальная идея была такова:
    Мета модель на классике + Laya(https://huggingface.co/convaiinnovations/laya), но временные рамки и вычислительные мощности ноутбука не позволяют локально гонять Laya, а kaggle и collab я не люблю особо
    """)
    return


@app.cell
def _(np, pl):
    features_train = pl.read_parquet("data/features_train.parquet")
    features_test = pl.read_parquet("data/features_test.parquet")

    train = pl.read_csv("data/train.csv")
    test = pl.read_csv("data/test.csv")

    features_train = features_train.join(
        train.select("cookie_id", "window_start_ts"), on="cookie_id", how="left"
    )
    # бьем дни на 7 фолдов
    days = features_train["window_start_ts"].unique().sort()
    day_groups = np.array_split(days.to_numpy(), 7)

    fold_map = {}
    for fold_id, group in enumerate(day_groups):
        for d in group:
            fold_map[d] = fold_id

    features_train = features_train.with_columns(
        pl.col("window_start_ts").replace(fold_map, return_dtype=pl.Int64).alias("fold")
    )

    features_train.group_by("fold").agg(pl.len().alias("n_cookies"), pl.col("target").mean().alias("bot_share")).sort("fold")
    return features_test, features_train


@app.cell
def _(features_train):
    feature_cols = [
        c for c in features_train.columns
        if c not in ("cookie_id", "target", "window_start_ts", "fold")
    ]
    len(feature_cols), feature_cols
    return (feature_cols,)


@app.cell
def _(CatBoostClassifier, feature_cols, features_train, np):
    X = features_train.select(feature_cols).to_pandas()
    y = features_train["target"].to_numpy()
    fold = features_train["fold"].to_numpy()

    oof = np.zeros(len(y))

    for f in range(7):
        is_valid = fold == f

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        model.fit(X[~is_valid], y[~is_valid])
        oof[is_valid] = model.predict_proba(X[is_valid])[:, 1]

    oof[:10]
    return X, fold, oof, y


@app.cell
def _(oof, precision_at_recall, y):
    score = precision_at_recall(y, oof)
    print("P@R>=0.7:", round(score, 4))

    # для сравнения уровень константы (если просто всем ставить одну оценку)
    print("доля ботов в train (уровень константы):", round(y.mean(), 4))
    return


@app.cell
def _(fold, oof, precision_at_recall, y):
    for q in range(7):
        is_val = fold == q
        fold_score = precision_at_recall(y[is_val], oof[is_val])
        print(f"фолд {q}: P@R>=0.7 = {round(fold_score, 4)}, кук = {is_val.sum()}, ботов = {y[is_val].sum()}")
    return


@app.cell
def _(CatBoostClassifier, X, y):
    final_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    final_model.fit(X, y)
    return (final_model,)


@app.cell
def _(feature_cols, features_test, final_model, pl):
    X_test = features_test.select(feature_cols).to_pandas()
    test_scores = final_model.predict_proba(X_test)[:, 1]

    submission = pl.DataFrame({
        "cookie_id": features_test["cookie_id"],
        "score": test_scores,
    })

    return (submission,)


@app.cell
def _(submission):
    submission.write_csv("submission.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Прогнал на степике, получил позорный результат. Попробую получить еще сильные фичи
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
