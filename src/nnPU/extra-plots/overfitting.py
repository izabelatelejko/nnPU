# %%
import os

import altair as alt
import pandas as pd
from save_chart import save_chart

test_metrics_per_epoch = pd.concat(
    [
        pd.read_json(
            os.path.join(
                "output",
                "[IMG] MNIST SS",
                "nnPUcc",
                "0.9",
                "100",
                "test_metrics_per_epoch.json",
            )
        ),
        pd.read_json(
            os.path.join(
                "output",
                "[IMG] MNIST SS",
                "nnPUss",
                "0.9",
                "100",
                "test_metrics_per_epoch.json",
            )
        ),
        pd.read_json(
            os.path.join(
                "output",
                "[IMG] Snacks SS",
                "nnPUcc",
                "0.9",
                "100",
                "test_metrics_per_epoch.json",
            )
        ),
        pd.read_json(
            os.path.join(
                "output",
                "[IMG] Snacks SS",
                "nnPUss",
                "0.9",
                "100",
                "test_metrics_per_epoch.json",
            )
        ),
        pd.read_json(
            os.path.join(
                "output",
                "[TXT] SMSSpam SS",
                "nnPUcc",
                "0.9",
                "100",
                "test_metrics_per_epoch.json",
            )
        ),
        pd.read_json(
            os.path.join(
                "output",
                "[TXT] SMSSpam SS",
                "nnPUss",
                "0.9",
                "100",
                "test_metrics_per_epoch.json",
            )
        ),
    ]
)
test_metrics_per_epoch = test_metrics_per_epoch.loc[
    test_metrics_per_epoch["epoch"] < 50
]
test_metrics_per_epoch

chart = (
    (
        alt.Chart(test_metrics_per_epoch, width=500, height=200)
        .mark_line()
        .encode(
            x=alt.X("epoch:Q").title("Epoch"),
            y=alt.Y("accuracy:Q").title("Accuracy"),
            color=alt.Color("model:N").legend(title="Method", labelLimit=400),
        )
        .facet(column=alt.Facet("dataset:N").title(None))
    )
    .configure_axis(labelFontSize=15, titleFontSize=15)
    .configure_header(titleFontSize=18, labelFontSize=18)
    .configure_legend(labelFontSize=15, titleFontSize=15)
)

save_chart(chart, "img", "overfitting")
chart

# %%
