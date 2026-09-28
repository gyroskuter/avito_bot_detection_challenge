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

    return (pl,)


@app.cell
def _(pl):
    ev = pl.read_csv('data/events.csv.gz')
    ev.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    в конце померяем все при помощи дистанционной корреляции(не забыть бы)
    """)
    return


@app.cell
def _(pl):
    train = pl.read_csv('data/train.csv')
    test = pl.read_csv('data/test.csv')
    events = pl.read_csv('data/events.csv.gz')


    ## дропнем дублирующую event_name колонку
    events = events.drop("eid")

    # убираем строки-дубликаты
    events = events.unique()

    # склеиваем метаданные train и test в одну табличку (у test просто будет target = null)
    cookies = pl.concat(
        [train, test.with_columns(pl.lit(None, dtype=pl.Int64).alias("target"))],
        how="diagonal",
    )

    events = events.join(
        cookies.select(["cookie_id", "window_start_ts", "window_end_ts", "target"]),
        on="cookie_id",
        how="inner",
    )

    # куки только в временном окне
    events = events.filter(
        (pl.col("event_ts") >= pl.col("window_start_ts"))
        & (pl.col("event_ts") < pl.col("window_end_ts"))
    )

    # приводим platform к человеческому виду
    events = events.with_columns(
        pl.col("platform").str.to_lowercase()
        .replace({"desktop": "web", "iphone": "ios"})
        .alias("platform_norm")
    )
    events = events.drop("platform")

    # event_ts сразу в datetime. т.к. дальше считаем интервалы, строки не подойдут
    events = events.with_columns(pl.col("event_ts").str.to_datetime())

    # сортировка для лагов
    events = events.sort(["cookie_id", "event_ts"])

    events = events.with_columns(
        (pl.col("event_ts") - pl.col("event_ts").shift(1).over("cookie_id"))
        .dt.total_seconds()
        .alias("gap")
    )

    events.select("cookie_id", "event_ts", "event_name", "gap").head(10)
    return cookies, events


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    сколько раз каждая кука делала каждый тип события
    """)
    return


@app.cell
def _(events, pl):
    event_counts = events.group_by(["cookie_id", "event_name"]).agg(
        pl.len().alias("cnt")
    ).pivot(index="cookie_id", on="event_name", values="cnt")

    event_counts = event_counts.fill_null(0)

    event_counts.head()
    return (event_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    сворачиваем события в одну строку на куку с временными признаками (сколько событий, ритм по gap), потом приклеиваем к этому счётчики по типам действий и превращаем их в доли от общего числа событий, чтобы сравнивать не абсолютную активность, а именно структуру поведения куки.
    """)
    return


@app.cell
def _(events, pl):
    # Блок 5 — базовые ритм-фичи
    base_features = events.group_by("cookie_id").agg(
        pl.len().alias("n_events"),
        pl.col("gap").median().alias("gap_median"),
        pl.col("gap").mean().alias("gap_mean"),
        pl.col("gap").std().alias("gap_std"),
        pl.col("gap").min().alias("gap_min"),
        pl.col("gap").max().alias("gap_max"),
        (pl.col("gap") <= 10).mean().alias("gap_short_share"),
        pl.col("target").first().alias("target"),
    )

    base_features.head()
    return (base_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    признаки по курсору(есть только на web)
    """)
    return


@app.cell
def _(events, pl):
    web_events = events.filter(
        (pl.col("platform_norm") == "web") & pl.col("pointer_x").is_not_null()
    )

    pointer_features = web_events.group_by("cookie_id").agg(
        pl.len().alias("ptr_n"),
        pl.col("pointer_x").std().alias("ptr_x_std"),
        pl.col("pointer_y").std().alias("ptr_y_std"),
        (pl.col("pointer_x").max() - pl.col("pointer_x").min()).alias("ptr_x_range"),
        (pl.col("pointer_y").max() - pl.col("pointer_y").min()).alias("ptr_y_range"),
    )

    pointer_features.head()
    return (pointer_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    у ботов курсов передается реже
    """)
    return


@app.cell
def _(events, pl):
    web_all = events.filter(pl.col("platform_norm") == "web")
    ptr_presence = web_all.group_by("cookie_id").agg(
        pl.col("pointer_x").is_not_null().mean().alias("ptr_present_share")
    )

    ptr_presence.head()
    return (ptr_presence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Глубина поиска
    """)
    return


@app.cell
def _(events, pl):
    search_events = events.filter(pl.col("event_name") == "search_results_view")

    search_features = search_events.group_by("cookie_id").agg(
        pl.col("search_page").max().alias("page_max"),
        pl.col("search_page").mean().alias("page_mean"),
        pl.col("search_query").n_unique().alias("query_nunique"),
    )

    search_features.head()
    return (search_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    разнообразие поиска
    """)
    return


@app.cell
def _(events, pl):
    diversity_features = events.group_by("cookie_id").agg(
        pl.col("item_id").n_unique().alias("item_nunique"),
        pl.col("item_category").n_unique().alias("category_nunique"),
        pl.col("item_location").n_unique().alias("location_nunique"),
    )

    diversity_features.head()
    return (diversity_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    смена User_agent
    """)
    return


@app.cell
def _(events, pl):
    events_ua = events.with_columns(
        (pl.col("user_agent") != pl.col("user_agent").shift(1).over("cookie_id"))
        .fill_null(False)
        .alias("ua_changed")
    )

    ua_features = events_ua.group_by("cookie_id").agg(
        pl.col("ua_changed").sum().alias("n_ua_changes")
    )

    ua_features.head()
    return (ua_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    возраст куки
    """)
    return


@app.cell
def _(cookies, pl):
    age_features = cookies.with_columns(
        (pl.col("window_start_ts").str.to_datetime() - pl.col("cookie_created_at").str.to_datetime())
        .dt.total_days()
        .alias("cookie_age_days")
    ).select("cookie_id", "cookie_age_days")

    age_features.head()
    return (age_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    target encoding по объявлениям - кодируем ботовость объявления без утечки.

    ставка куки по объявлению = доля ботов среди ДРУГИХ кук, смотревших это же объявление,
    кроме кук из её же дня (иначе куки одного дня узнают таргет друг друга через день целиком).
    для test всегда берём статистику по всему train - test дни строго позже, это честно
    """)
    return


@app.cell
def _(events, pl):
    item_pairs = (
        events.filter(pl.col("item_id").is_not_null())
        .unique(subset=["cookie_id", "item_id"])
        .select("cookie_id", "item_id", "window_start_ts", "target")
    )
    item_pairs.head()
    return (item_pairs,)


@app.cell
def _(item_pairs, pl):
    # статистика только по размеченным кукам (test сюда не попадает)
    item_pairs_labeled = item_pairs.filter(pl.col("target").is_not_null())

    item_totals = item_pairs_labeled.group_by("item_id").agg(
        pl.col("target").sum().alias("total_sum"),
        pl.len().alias("total_n"),
    )
    item_by_day = item_pairs_labeled.group_by(["item_id", "window_start_ts"]).agg(
        pl.col("target").sum().alias("day_sum"),
        pl.len().alias("day_n"),
    )
    item_by_day.head()
    return item_by_day, item_pairs_labeled, item_totals


@app.cell
def _(cookies, item_by_day, item_totals, pl):
    # base rate по кукам, а не по парам кука+объявление - иначе активные боты (у них больше объявлений)
    # перевесят пары и prior получится завышенным
    prior = cookies.filter(pl.col("target").is_not_null())["target"].mean()
    prior_weight = 10  # сглаживание к базовой доле, чтобы объявление с 1-2 просмотрами не давало 0% или 100%

    # train: вычитаем свой день из общей статистики
    item_oof = item_by_day.join(item_totals, on="item_id", how="left").with_columns(
        ((pl.col("total_sum") - pl.col("day_sum") + prior_weight * prior) /
         (pl.col("total_n") - pl.col("day_n") + prior_weight)).alias("item_rate")
    ).select("item_id", "window_start_ts", "item_rate")

    # test: берём всё train целиком, вычитать нечего
    item_full = item_totals.with_columns(
        ((pl.col("total_sum") + prior_weight * prior) / (pl.col("total_n") + prior_weight)).alias("item_rate")
    ).select("item_id", "item_rate")

    print("базовая доля ботов (prior):", prior)
    item_oof.head()
    return item_full, item_oof, prior


@app.cell
def _(item_full, item_oof, item_pairs, pl, prior):
    pairs_train = item_pairs.filter(pl.col("target").is_not_null()).join(
        item_oof, on=["item_id", "window_start_ts"], how="left"
    )
    pairs_test = item_pairs.filter(pl.col("target").is_null()).join(
        item_full, on="item_id", how="left"
    )
    # объявление, которого train вообще не видел - просто базовая доля, ничего не знаем
    item_pairs_rated = pl.concat([pairs_train, pairs_test], how="diagonal").with_columns(
        pl.col("item_rate").fill_null(prior)
    )
    item_pairs_rated.head()
    return (item_pairs_rated,)


@app.cell
def _(item_pairs_rated, pl):
    item_te = item_pairs_rated.group_by("cookie_id").agg(
        pl.col("item_rate").mean().alias("item_te_mean"),
        pl.col("item_rate").max().alias("item_te_max"),
    )
    item_te.head()
    return (item_te,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    то же самое для связки запрос+регион - паттерн один в один, просто другой ключ вместо item_id
    """)
    return


@app.cell
def _(events, pl):
    qloc_pairs = (
        events.filter(pl.col("event_name") == "search_results_view")
        .with_columns((pl.col("search_query") + "|" + pl.col("item_location")).alias("qloc"))
        .unique(subset=["cookie_id", "qloc"])
        .select("cookie_id", "qloc", "window_start_ts", "target")
    )
    qloc_pairs.head()
    return (qloc_pairs,)


@app.cell
def _(pl, qloc_pairs):
    qloc_pairs_labeled = qloc_pairs.filter(pl.col("target").is_not_null())

    qloc_totals = qloc_pairs_labeled.group_by("qloc").agg(
        pl.col("target").sum().alias("total_sum"),
        pl.len().alias("total_n"),
    )
    qloc_by_day = qloc_pairs_labeled.group_by(["qloc", "window_start_ts"]).agg(
        pl.col("target").sum().alias("day_sum"),
        pl.len().alias("day_n"),
    )
    qloc_by_day.head()
    return qloc_by_day, qloc_pairs_labeled, qloc_totals


@app.cell
def _(cookies, pl, qloc_by_day, qloc_totals):
    # тот же фикс, что и для item_te: prior по кукам, а не по парам кука+связка
    qloc_prior = cookies.filter(pl.col("target").is_not_null())["target"].mean()
    qloc_prior_weight = 10

    qloc_oof = qloc_by_day.join(qloc_totals, on="qloc", how="left").with_columns(
        ((pl.col("total_sum") - pl.col("day_sum") + qloc_prior_weight * qloc_prior) /
         (pl.col("total_n") - pl.col("day_n") + qloc_prior_weight)).alias("qloc_rate")
    ).select("qloc", "window_start_ts", "qloc_rate")

    qloc_full = qloc_totals.with_columns(
        ((pl.col("total_sum") + qloc_prior_weight * qloc_prior) / (pl.col("total_n") + qloc_prior_weight)).alias("qloc_rate")
    ).select("qloc", "qloc_rate")

    qloc_oof.head()
    return qloc_full, qloc_oof, qloc_prior


@app.cell
def _(pl, qloc_full, qloc_oof, qloc_pairs, qloc_prior):
    qloc_pairs_train = qloc_pairs.filter(pl.col("target").is_not_null()).join(
        qloc_oof, on=["qloc", "window_start_ts"], how="left"
    )
    qloc_pairs_test = qloc_pairs.filter(pl.col("target").is_null()).join(
        qloc_full, on="qloc", how="left"
    )
    qloc_pairs_rated = pl.concat([qloc_pairs_train, qloc_pairs_test], how="diagonal").with_columns(
        pl.col("qloc_rate").fill_null(qloc_prior)
    )
    qloc_pairs_rated.head()
    return (qloc_pairs_rated,)


@app.cell
def _(pl, qloc_pairs_rated):
    qloc_te = qloc_pairs_rated.group_by("cookie_id").agg(
        pl.col("qloc_rate").mean().alias("qloc_te_mean"),
        pl.col("qloc_rate").max().alias("qloc_te_max"),
    )
    qloc_te.head()
    return (qloc_te,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    финальная сборка
    """)
    return


@app.cell
def _(
    age_features,
    base_features,
    diversity_features,
    event_counts,
    item_te,
    pl,
    pointer_features,
    ptr_presence,
    qloc_te,
    search_features,
    ua_features,
):
    features = (
        base_features
        .join(event_counts, on="cookie_id", how="left")
        .join(pointer_features, on="cookie_id", how="left")
        .join(ptr_presence, on="cookie_id", how="left")
        .join(search_features, on="cookie_id", how="left")
        .join(diversity_features, on="cookie_id", how="left")
        .join(ua_features, on="cookie_id", how="left")
        .join(age_features, on="cookie_id", how="left")
        .join(item_te, on="cookie_id", how="left")
        .join(qloc_te, on="cookie_id", how="left")
    )

    event_cols = [c for c in event_counts.columns if c != "cookie_id"]
    features = features.with_columns(
        [(pl.col(c) / pl.col("n_events")).alias(f"share_{c}") for c in event_cols]
    )

    features.head()
    return (features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Выгружаем полученные фичи
    """)
    return


@app.cell
def _(features, pl):
    features_train = features.filter(pl.col("target").is_not_null())
    features_test = features.filter(pl.col("target").is_null()).drop("target")

    features_train.write_parquet("data/features_train.parquet")
    features_test.write_parquet("data/features_test.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
