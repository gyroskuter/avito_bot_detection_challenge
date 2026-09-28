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
    import polars as pl
    import matplotlib.pyplot as plt
    pl.Config.set_tbl_rows(-1)
    return pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Будущие коллеги. Небольшое объявление:
    1) я работаю в маримо ноутбуке, что накладывает определенные ограничения на названия переменных и порядок выполнения операций

    2) возьмите меня на работу пж.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3) мне надо кормить эту девчонку
    ![alt](public/cat.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Читаем данные
    """)
    return


@app.cell
def _(pl):
    train = pl.read_csv('data/train.csv')
    test = pl.read_csv('data/test.csv')
    return (train,)


@app.cell
def _(pl, train):
    events = pl.read_csv('data/events.csv.gz')

    #дропнем дублирующие друг друга колонки (eid-event_name)
    events = events.drop("eid")

    # убираем строки-дубликаты
    events = events.unique()  

    # прикрепим таргеты из трейна(тест прикреплять не будем, это утечка)
    events = events.join(
        train.select(["cookie_id", "window_start_ts", "window_end_ts", "target"]),
        on="cookie_id",
        how="inner",  
    )

    # куки только в временном окне
    events = events.filter(
        (pl.col("event_ts") >= pl.col("window_start_ts"))
        & (pl.col("event_ts") < pl.col("window_end_ts"))
    )

    # на самом деле можно и убрать фичу captcha_shown, т.к. мы полагаемся тут на другой сервис, который определяет капчу, но это более философский вопрос

    # приводим колонку к человеческому виду
    events = events.with_columns(
        pl.col("platform").str.to_lowercase()
        .replace({"desktop": "web", "iphone": "ios"})
        .alias("platform_norm")
    )
    events = events.drop("platform")
    return (events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Посмотрим дисбаланс классов
    """)
    return


@app.cell
def _(events, pl):
    print("1 percent: ", events.filter(pl.col("target") == 1).height / events["target"].shape[0])
    #Видно сильный дисбаланс классов, на него будем немного опираться при выборе фичей в дальнейшем. 
    #p.s. понятно, что это дисбаланс по событиям, а не по кукам, но says a lot about society
    return


@app.cell
def _(events):
    events.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    попробуем посмотреть закономерности между разными значениями по фичам и таргетом == 1
    """)
    return


@app.cell
def _(events, pl, plt):
    cat_cols = ["event_name", "platform_norm", "seller_type", "item_category", "item_location", "search_query", "seller_type", "search_page"]

    heights = [events[col].n_unique() for col in cat_cols]
    fig, axes = plt.subplots(len(cat_cols), 1, figsize=(7, sum(0.35 * h + 1 for h in heights)))

    for ax, col in zip(axes, cat_cols):
        stats = (
            events.filter(pl.col("target").is_not_null())
            .with_columns(pl.col(col).fill_null("missing"))
            .group_by(col)
            .agg(pl.col("target").mean().alias("bot_share"))
            .sort("bot_share")
        )
        ax.barh(stats[col].to_list(), stats["bot_share"].to_list(), color="gray")
        ax.set_title(col)
        ax.set_xlabel("доля ботов")
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *Вывод:*

    1) боты "копают" глубже(смотрят больше страниц)

    2) в web сильно больше ботов

    3) можно понавешать коэффициентов на serch_query

    4) показатели и разница в item_name для меня несущественна, чтобы трогать ее
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Попробуем посмотреть на user_agent и изучить его
    """)
    return


@app.cell
def _(events):
    events["user_agent"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    хммм, пайтон реквест, хмммм....
    """)
    return


@app.cell
def _(events, pl):
    events.group_by("user_agent").agg(pl.col("target").mean().alias("target_pct"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Вывод:
    как будто логично выделить топовые по этой табличке user_agent как фичу
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Гипотеза: смена user_agent свойственна человеку, а не боту.
    """)
    return


@app.cell
def _(events):
    events.head()
    return


@app.cell
def _(events, pl):
    events_ua = events.sort(["cookie_id", "event_ts"]).with_columns(
        (pl.col("user_agent") != pl.col("user_agent").shift(1).over("cookie_id"))
        .fill_null(False)
        .alias("ua_changed")
    )
    events_ua.head()
    return (events_ua,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    сворачиваем до одной строки на куку сколько раз менялся UA, и смотрим долю ботов по группам.
    """)
    return


@app.cell
def _(events_ua, pl):
    cookie_ua = events_ua.group_by("cookie_id").agg(
        pl.col("ua_changed").sum().alias("n_ua_changes"),
        pl.col("target").first().alias("target"),
    )

    cookie_ua.group_by("n_ua_changes").agg(
        pl.len().alias("n_cookies"),
        (pl.col("target").mean() * 100).alias("bot_pct"),
    ).sort("n_ua_changes")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Вывод:

    1) анлак, гипотеза не подтвердилась
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Гипотеза: люди действуют более медленно, чем боты
    """)
    return


@app.cell
def _(events_ua, pl):
    events_gap = events_ua.with_columns(
        pl.col("event_ts").str.to_datetime()
    ).sort(["cookie_id", "event_ts"]).with_columns(
        (pl.col("event_ts") - pl.col("event_ts").shift(1).over("cookie_id"))
        .dt.total_seconds()
        .alias("gap")
    )
    events_gap.select("cookie_id", "event_ts", "gap").head(10)
    return (events_gap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    сворачиваем до куки: медианный gap и доля коротких интервалов (<=10 сек).
    опять же считаем по кукам, а не по событиям - причина та же, что и с UA
    """)
    return


@app.cell
def _(events_gap, pl):
    cookie_gap = events_gap.group_by("cookie_id").agg(
        pl.col("gap").median().alias("gap_median"),
        (pl.col("gap") <= 10).mean().alias("gap_short_share"),
        pl.col("target").first().alias("target"),
    )
    cookie_gap.describe()
    return (cookie_gap,)


@app.cell
def _(cookie_gap, pl):
    (
        cookie_gap.with_columns(
            pl.col("gap_median").cut([5, 10, 20, 45, 120, 300], left_closed=True).alias("gap_bucket")
        )
        .group_by("gap_bucket")
        .agg(pl.len().alias("n_cookies"), (pl.col("target").mean() * 100).alias("bot_pct"))
        .sort("gap_bucket")
    )
    return


@app.cell
def _(cookie_gap, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for _t, _lbl in [(0, "люди"), (1, "боты")]:
        _sub = cookie_gap.filter(cookie_gap["target"] == _t)
        _ax[0].hist(_sub["gap_median"].drop_nulls().clip(upper_bound=300), bins=40, density=True, alpha=0.55, label=_lbl, color="gray" if _t == 0 else "black")
        _ax[1].hist(_sub["gap_short_share"].drop_nulls(), bins=30, density=True, alpha=0.55, label=_lbl, color="gray" if _t == 0 else "black")
    _ax[0].set_title("медианный интервал по куке")
    _ax[1].set_title("доля интервалов <=10 сек по куке")
    _ax[0].legend()
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Вывод:

    1) кука)

    1) ритм - отличный сигнал, смело добавляем как фичу

    2) много фич со временными метками
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Гипотеза: курсор у ботов почти не двигается и передаётся реже
    """)
    return


@app.cell
def _(events_gap, pl):
    cookie_pointer = (
        events_gap.filter((pl.col("platform_norm") == "web") & pl.col("pointer_x").is_not_null())
        .group_by("cookie_id")
        .agg(pl.col("pointer_x").std().alias("ptr_x_std"), pl.col("target").first().alias("target"))
    )

    ptr_presence = (
        events_gap.filter(pl.col("platform_norm") == "web")
        .group_by("cookie_id")
        .agg(pl.col("pointer_x").is_not_null().mean().alias("ptr_present_share"), pl.col("target").first().alias("target"))
    )

    print("corr ptr_x_std ~ target:", cookie_pointer.select(pl.corr("ptr_x_std", "target")).item())
    print("corr ptr_present_share ~ target:", ptr_presence.select(pl.corr("ptr_present_share", "target")).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # гипотеза подтвердилась. курсор берем в фичи
    у ботов курсор и передаётся реже, а когда передаётся, то почти не двигается
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    гипотеза: боты смотрят много объявлений в разных регионах, но не скачут по категориям
    """)
    return


@app.cell
def _(events_gap, pl):
    diversity = events_gap.group_by("cookie_id").agg(
        pl.col("item_id").n_unique().alias("item_nunique"),
        pl.col("item_category").n_unique().alias("category_nunique"),
        pl.col("item_location").n_unique().alias("location_nunique"),
        pl.col("search_query").n_unique().alias("query_nunique"),
        pl.col("target").first().alias("target"),
    )

    for c in ["item_nunique", "category_nunique", "location_nunique", "query_nunique"]:
        print(f"corr {c} ~ target:", diversity.select(pl.corr(c, "target")).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Итоговый вывод

    1) вообще задача поставлена слегка некорректно по моему скромному мнению. тут нужно угадать, кого внешний или внутренний сервис по отлову ботов считает ботом

    2) сильный дисбаланс

    3) боты копают глубже по страницам

    4) web более более ботовый

    5) user_agent достаточно важный признак

    6) на скорости взаимодействия бота можно построить кучу фич по идее

    7) капча как признак бесполезна + как я говорил, можно посчитать утечкой, но она всегда за интересующим нас окном

    8) курсор мыши хорошая фича

    9) сколько объявлений и регионов смотрит кука тоже стоит добавить, а разнообразие категорий отстой
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    исследование доп фичи/фичей после результата на степике
    гипотеза: боты ходят по одним и тем же объявлениям или связкам запрос+регион
    """)
    return


@app.cell
def _(events_gap, pl):
    it = (
        events_gap.filter(pl.col("item_id").is_not_null() & pl.col("target").is_not_null())
        .unique(subset=["cookie_id", "item_id"])
    )
    item_stats = it.group_by("item_id").agg(pl.len().alias("n"), pl.col("target").mean().alias("bot_share"))
    item_stats5 = item_stats.filter(pl.col("n") >= 5)

    print("объявлений с n>=5 кук:", item_stats5.height)
    item_stats5["bot_share"].describe()
    return (item_stats5,)


@app.cell
def _(events_gap, item_stats5, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.hist(item_stats5["bot_share"].to_list(), bins=30, color="gray")
    _ax.axvline(events_gap.unique(subset="cookie_id")["target"].mean(), color="black", linestyle="--", label="бейз")
    _ax.set_title("доля ботов среди кук, смотревших объявление (n>=5)")
    _ax.legend()
    _fig
    return


@app.cell
def _(events_gap, pl):
    s = (
        events_gap.filter((pl.col("event_name") == "search_results_view") & pl.col("target").is_not_null())
        .with_columns((pl.col("search_query") + "|" + pl.col("item_location")).alias("qloc"))
    )
    qloc_stats = s.unique(subset=["cookie_id", "qloc"]).group_by("qloc").agg(
        pl.len().alias("n"), pl.col("target").mean().alias("bot_share")
    )
    qloc_stats5 = qloc_stats.filter(pl.col("n") >= 5)

    qloc_stats5["bot_share"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## вывод: явная кластеризация ботов по конкретным объектам. можно кодировать "ботовость" объекта как фичу -
    но ТОЛЬКО через OOF
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
