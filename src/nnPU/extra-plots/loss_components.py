# %%
import os

import altair as alt
import pandas as pd
from save_chart import save_chart

loss_history = pd.concat(
    [
        pd.read_csv(
            os.path.join(
                "output",
                "[IMG] Snacks SS",
                "nnPUcc",
                "0.9",
                "100",
                "loss-history.csv",
            )
        )
        .melt(id_vars="Epoch", var_name="Component", value_name="Value")
        .assign(Model="nnPUcc", Dataset="[IMG] Snacks SS"),
        pd.read_csv(
            os.path.join(
                "output",
                "[IMG] Snacks SS",
                "nnPUss",
                "0.9",
                "100",
                "loss-history.csv",
            )
        )
        .melt(id_vars="Epoch", var_name="Component", value_name="Value")
        .assign(Model="nnPUss", Dataset="[IMG] Snacks SS"),
        pd.read_csv(
            os.path.join(
                "output",
                "[IMG] Snacks CC",
                "nnPUcc",
                "0.9",
                "100",
                "loss-history.csv",
            )
        )
        .melt(id_vars="Epoch", var_name="Component", value_name="Value")
        .assign(Model="nnPUcc", Dataset="[IMG] Snacks CC"),
        pd.read_csv(
            os.path.join(
                "output",
                "[IMG] Snacks CC",
                "nnPUss",
                "0.9",
                "100",
                "loss-history.csv",
            )
        )
        .melt(id_vars="Epoch", var_name="Component", value_name="Value")
        .assign(Model="nnPUss", Dataset="[IMG] Snacks CC"),
    ]
)
loss_history

import numpy as np

loss_history_pivot = loss_history.pivot(
    index=["Model", "Dataset", "Epoch"], columns="Component", values="Value"
).reset_index(drop=False)


loss_history_pivot["Model Rᴰ"] = np.where(
    loss_history_pivot["Model"].str.contains("nnPUss"),
    loss_history_pivot["Whole distribution component SS"],
    loss_history_pivot["Whole distribution component CC"],
)
loss_history_pivot["Correct Rᴰ"] = np.where(
    loss_history_pivot["Dataset"].str.contains(" SS"),
    loss_history_pivot["Whole distribution component SS"],
    loss_history_pivot["Whole distribution component CC"],
)

loss_history_pivot["Model Rᴰ - Rᶜᵒʳʳ"] = np.where(
    loss_history_pivot["Model"].str.contains("nnPUss"),
    loss_history_pivot["Whole distribution component SS"]
    - loss_history_pivot["PU SCAR correction"],
    loss_history_pivot["Whole distribution component CC"]
    - loss_history_pivot["PU SCAR correction"],
)
loss_history_pivot["Correct Rᴰ - Rᶜᵒʳʳ"] = np.where(
    loss_history_pivot["Dataset"].str.contains(" SS"),
    loss_history_pivot["Whole distribution component SS"]
    - loss_history_pivot["PU SCAR correction"],
    loss_history_pivot["Whole distribution component CC"]
    - loss_history_pivot["PU SCAR correction"],
)

loss_history_pivot["Model R"] = np.where(
    loss_history_pivot["Model"].str.contains("nnPUss"),
    loss_history_pivot["Labeled component"]
    + loss_history_pivot["Whole distribution component SS"]
    - loss_history_pivot["PU SCAR correction"],
    loss_history_pivot["Labeled component"]
    + loss_history_pivot["Whole distribution component CC"]
    - loss_history_pivot["PU SCAR correction"],
)
loss_history_pivot["Correct R"] = np.where(
    loss_history_pivot["Dataset"].str.contains(" SS"),
    loss_history_pivot["Labeled component"]
    + loss_history_pivot["Whole distribution component SS"]
    - loss_history_pivot["PU SCAR correction"],
    loss_history_pivot["Labeled component"]
    + loss_history_pivot["Whole distribution component CC"]
    - loss_history_pivot["PU SCAR correction"],
)

loss_history_pivot["Rᴸ"] = loss_history_pivot["Labeled component"]
loss_history_pivot["Rᶜᵒʳʳ"] = loss_history_pivot["PU SCAR correction"]

loss_history_pivot = loss_history_pivot.drop(
    columns=[
        "Calculated loss",
        "Labeled component",
        "Whole distribution component SS",
        "Whole distribution component CC",
        "PU SCAR correction",
    ]
)

loss_history = loss_history_pivot.melt(
    id_vars=["Model", "Dataset", "Epoch"], var_name="Component", value_name="Value"
)

loss_history["Component type"] = np.where(
    loss_history["Component"].str.contains("Correct"), "Correct value", "Method value"
)
loss_history["Component"] = np.where(
    loss_history["Component"].str.contains(" "),
    loss_history["Component"].str.split(" ").apply(lambda arr: " ".join(arr[1:])),
    loss_history["Component"],
)


components = [
    "Rᴸ",
    # "Model Rᴰ",
    # "Correct Rᴰ",
    # "Model Rᴰ - Rᶜᵒʳʳ",
    # "Correct Rᴰ - Rᶜᵒʳʳ",
    "Rᴰ",
    # "Rᴰ",
    "Rᴰ - Rᶜᵒʳʳ",
    # "Rᴰ - Rᶜᵒʳʳ",
    "Rᶜᵒʳʳ",
    "R",
    # "Model R",
    # "Correct R",
]
colors = [
    "green",
    # "red",
    "red",
    # "orange",
    "orange",
    "grey",
    # "black",
    "black",
]


chart = (
    (
        alt.Chart(loss_history, width=500, height=200)
        .mark_line(opacity=0.7)
        .encode(
            x=alt.X("Epoch:Q"),
            y=alt.Y("Value:Q"),
            color=alt.Color("Component:N").scale(domain=components, range=colors),
            strokeDash=alt.StrokeDash("Component type:N").sort(
                ["Method value", "Correct value"]
            ),
            # shape=alt.Shape("Component:N"),
        )
        .facet(
            column=alt.Facet("Model:N").title(None),
            row=alt.Facet("Dataset:N").title(None),
        )
    )
    .configure_point(size=10)
    .configure_axis(labelFontSize=15, titleFontSize=15)
    .configure_header(titleFontSize=18, labelFontSize=18)
    .configure_legend(labelFontSize=15, titleFontSize=15)
)

save_chart(chart, "img", "loss_components")
chart

# %%
