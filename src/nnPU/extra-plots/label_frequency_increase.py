# %%
import json
import os

import pandas as pd
from IPython.display import display

RESULTS_DIR = "output"

results = []
for directory, subdirs, files in os.walk(RESULTS_DIR):
    if "metrics.json" in files:
        metrics_file_path = os.path.join(directory, "metrics.json")
        with open(metrics_file_path, "r") as f:
            metric_values = json.load(f)
            if metric_values["label_frequency"] == 0.02:
                continue

            metric_values["label_frequency"] = f'{metric_values["label_frequency"]:.1f}'
            results.append(metric_values)
        # print(metrics_file_path)

results_df = pd.DataFrame.from_records(results)
display(results_df)

# %%
import altair as alt
from save_chart import save_chart

mean_results_df = (
    results_df.groupby(["dataset", "label_frequency", "model"])
    .mean()
    .reset_index(drop=False)
)

mean_results_ss_df = mean_results_df.loc[mean_results_df["dataset"].str.contains(" SS")]
mean_results_cc_df = mean_results_df.loc[mean_results_df["dataset"].str.contains(" CC")]

chart = (
    alt.Chart(mean_results_ss_df, width=200, height=120)
    .mark_line()
    .encode(
        x=alt.X("label_frequency").title("Label frequency"),
        y=alt.Y("accuracy").scale(zero=False).title("Accuracy"),
        color=alt.Color("dataset").legend(None),
        strokeDash=alt.StrokeDash("model").sort(["nnPUss", "nnPUcc"]).title("Method"),
        facet=alt.Facet("dataset").columns(6).title(None),
    )
    .configure_axis(labelFontSize=14, titleFontSize=14)
    .configure_header(titleFontSize=16, labelFontSize=16)
    .configure_legend(labelFontSize=14, titleFontSize=14)
)
save_chart(chart, "img", "lf_increase_ss")
chart

# %%
chart = (
    alt.Chart(mean_results_cc_df, width=200, height=120)
    .mark_line()
    .encode(
        x=alt.X("label_frequency").title("Label frequency"),
        y=alt.Y("accuracy").scale(zero=False).title("Accuracy"),
        color=alt.Color("dataset").legend(None),
        strokeDash=alt.StrokeDash("model").sort(["nnPUcc", "nnPUss"]).title("Method"),
        facet=alt.Facet("dataset").columns(6).title(None),
    )
    .configure_axis(labelFontSize=14, titleFontSize=14)
    .configure_header(titleFontSize=16, labelFontSize=16)
    .configure_legend(labelFontSize=14, titleFontSize=14)
)
save_chart(chart, "img", "lf_increase_cc")
chart

# %%
