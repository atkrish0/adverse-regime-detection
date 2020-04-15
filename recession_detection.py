import datetime
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, preprocessing, linear_model, pipeline

start = "1974-01-01"
end = datetime.date.today()


def pct_change_on_last_year(df):
    "compute pct_change on previous year, assuming quarterly"
    return (df - df.shift(4))/df.shift(4)


def get_indicators_from_fred(start=start, end=end):
    """
    Fetch quarterly data on 6 leading indicators from time period start:end
    """
    # yield curve, unemployment, change in inventory, new private housing permits
    yc_unemp_inv_permit = (
        web.DataReader(["T10Y2Y", "UNRATE", "CBIC1",
                        "PERMIT"], "fred", start, end)
        .resample("QS")
        .mean()
    )

    # percent change in housing prices and retail sales
    hpi_retail = (
        web.DataReader(["USSTHPI", "SLRTTO01USQ661S"], "fred", start, end)
        .resample("QS")  # already quarterly, adjusting so index is same
        .mean()
        .pipe(pct_change_on_last_year)
        .dropna()
    )

    indicators = (
        yc_unemp_inv_permit
        .join(hpi_retail)
        .dropna()
        .rename(columns=dict(
            USSTHPI="pct_change_hpi",
            T10Y2Y="yield_curve",
            UNRATE="unemp",
            CBIC1="inventory",
            SLRTTO01USQ661S="retail_sales",
            PERMIT="house_permits"
        ))
    )

    return indicators


indicators = get_indicators_from_fred()


def get_recession_data():
    recession = (
        web.DataReader(["USRECQ"], "fred", start, end)
        .rename(columns=dict(USRECQ="recession"))
        ["recession"]
    )

    # extract start and end date for each recession
    start_dates = recession.loc[recession.diff() > 0].index.tolist()
    if recession.iloc[0] > 0:
        start_dates = [recession.index[0]] + start_dates

    end_dates = recession.loc[recession.diff() < 0].index.tolist()

    if len(start_dates) != len(end_dates):
        raise ValueError("Need to have same number of start/end dates!")

    return recession, start_dates, end_dates


recession, start_dates, end_dates = get_recession_data()


def add_recession_bands(ax):
    for s, e in zip(start_dates, end_dates):
        ax.axvspan(s, e, color="grey", alpha=0.2)


axs = indicators.plot(subplots=True, figsize=(8, 6),
                      layout=(3, 2), legend=False)

for i, ax in enumerate(axs.flatten()):
    add_recession_bands(ax)
    ax.set_title(list(indicators)[i])

fig = axs[0, 0].get_figure()
fig.tight_layout()


def make_train_data(indicators, rec, nlead=4):
    return indicators.join(rec.shift(nlead)).dropna()


def fit_for_nlead(ind, rec, nlead, mod):
    df = make_train_data(ind, rec, nlead)
    X = df.drop(["recession"], axis=1).copy()
    y = df["recession"].copy()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    mod.fit(X_train, y_train)
    cmat = metrics.confusion_matrix(y_test, mod.predict(X_test))
    return cmat


mod = pipeline.make_pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression(solver="lbfgs")
)

cmats = dict()
for nlead in range(1, 11):
    cmats[nlead] = np.zeros((2, 2))
    print(f"starting for {nlead} leads")
    for rep in range(200):
        cmats[nlead] += fit_for_nlead(indicators, recession, nlead, mod)

    cmats[nlead] = cmats[nlead] / 200

for k, v in cmats.items():
    print(f"\n\nThe average confusion matrix for {k} lag(s) was:\n {v}")
