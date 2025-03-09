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

results_df = pd.DataFrame.from_records(results)
display(results_df)

# %%
pd.set_option("display.max_rows", 500)
(
    results_df[results_df.dataset.str.contains(" SS")].pivot_table(
        values="accuracy",
        index=["dataset", "label_frequency"],
        columns="model",
        aggfunc=len,
    )
)

# %%
for metric in ["accuracy", "precision", "recall", "f1"]:
    df = (
        results_df[results_df.dataset.str.contains(" CC")].pivot_table(
            values=metric, index=["dataset", "label_frequency"], columns="model"
        )
        * 100
    )
    df["nnPUcc improvement"] = df["nnPUcc"] - df["nnPUss"]
    # df["uPUcc improvement"] = df["uPUcc"] - df["uPUss"]
    df = df.round(2)
    df.to_csv(f"csv/CC-datasets-{metric}.csv")
    diff_pivot = df.reset_index(drop=False).pivot_table(
        values="nnPUcc improvement",
        index="label_frequency",
        columns="dataset",
    )
    diff_pivot.to_csv(f"csv/CC-datasets-{metric}-diff-pivot.csv")
    if metric in ["accuracy"]:
        display(df)
        display(diff_pivot)

# %%
for metric in ["accuracy", "precision", "recall", "f1"]:
    df = (
        results_df[results_df.dataset.str.contains(" SS")].pivot_table(
            values=metric, index=["dataset", "label_frequency"], columns="model"
        )
        * 100
    )
    df["nnPUss improvement"] = df["nnPUss"] - df["nnPUcc"]
    # df["uPUss improvement"] = df["uPUss"] - df["uPUcc"]
    df = df.round(2)
    df.to_csv(f"csv/SS-datasets-{metric}.csv")
    diff_pivot = df.reset_index(drop=False).pivot_table(
        values="nnPUss improvement",
        index="label_frequency",
        columns="dataset",
    )
    diff_pivot.to_csv(f"csv/SS-datasets-{metric}-diff-pivot.csv")
    if metric in ["accuracy"]:
        display(df)
        display(diff_pivot)


# %%
import re

DIR = "latex"


def merge_latex_headers(latex_table, scaling=None):
    table_lines = latex_table.split("\n")
    tabular_start = 0
    tabular_end = len(table_lines) - 3

    def process_line(l):
        return [
            "\\textbf{" + name.replace("\\", "").strip() + "}"
            for name in l.split("&")
            if name.replace("\\", "").strip() != ""
        ]

    header_line, index_line = (
        table_lines[tabular_start + 1],
        table_lines[tabular_start + 2],
    )
    headers = process_line(header_line)
    index_names = process_line(index_line)

    new_headers = index_names + headers
    new_headers = " & ".join(new_headers) + " \\\\"

    table_lines.remove(header_line)
    table_lines.remove(index_line)
    table_lines.insert(tabular_start + 1, new_headers)

    table_lines = [
        "\t" + l if i > tabular_start and i < tabular_end else l
        for i, l in enumerate(table_lines)
    ]

    table_lines.insert(tabular_end, "\\bottomrule")

    inserted_extra_separators = 0
    row = tabular_end - 3
    while row > tabular_start + 2:
        table_lines.insert(row, "\\midrule")
        inserted_extra_separators += 1
        row -= 3

    table_lines.insert(tabular_start + 2, "\\midrule")
    table_lines.insert(tabular_start + 1, "\\toprule")

    tabular_end += 3 + inserted_extra_separators

    if scaling is not None:
        table_lines.insert(tabular_end + 1, "}")
        table_lines.insert(tabular_start, "\scalebox{" + f"{scaling:.2f}" + "}{")

    return "\n".join(table_lines)


for metric in ["accuracy", "precision", "recall", "f1"]:
    df = results_df[results_df.dataset.str.contains(" CC")].pivot_table(
        values=metric, index=["dataset", "label_frequency"], columns="model"
    )
    df["$\Delta$"] = df["nnPUcc"] - df["nnPUss"]
    df = df.reset_index(drop=False).melt(
        id_vars=["dataset", "label_frequency"], value_name=metric
    )
    df.model = pd.Categorical(
        df.model, categories=["nnPUss", "nnPUcc", "$\Delta$"], ordered=True
    )
    df = (
        df.pivot_table(
            values=metric, index=["label_frequency", "model"], columns=["dataset"]
        )
        * 100
    )

    df = df.round(2)
    df.to_csv(f"csv/CC-datasets-{metric}.csv")

    df.columns.name = None
    df.index.names = ["c", "Model"]
    df.columns = [
        col.replace(" CC", "")
        .replace(" SS", "")
        .replace("[IMG] ", "")
        .replace("[TAB] ", "")
        .replace("[TXT] ", "")
        for col in df.columns
    ]
    if metric in ["accuracy"]:
        display(df)

    os.makedirs(DIR, exist_ok=True)
    for i, df_half in enumerate(
        [
            df.iloc[:, : len(df.columns) // 2],
            df.iloc[:, len(df.columns) // 2 :],
        ]
    ):
        latex_table = df_half.style.format(precision=2).to_latex(
            column_format="l|c|" + "c" * len(df_half.columns)
        )
        latex_table = merge_latex_headers(latex_table, scaling=0.75)
        with open(os.path.join(DIR, f"CC-{metric}-p{i+1}.tex"), "w") as f:
            f.write(latex_table)

# %%
DIR = "latex"

for metric in ["accuracy", "precision", "recall", "f1"]:
    df = results_df[results_df.dataset.str.contains(" SS")].pivot_table(
        values=metric, index=["dataset", "label_frequency"], columns="model"
    )
    df["$\Delta$"] = df["nnPUss"] - df["nnPUcc"]
    df = df.reset_index(drop=False).melt(
        id_vars=["dataset", "label_frequency"], value_name=metric
    )
    df.model = pd.Categorical(
        df.model, categories=["nnPUcc", "nnPUss", "$\Delta$"], ordered=True
    )
    df = (
        df.pivot_table(
            values=metric, index=["label_frequency", "model"], columns=["dataset"]
        )
        * 100
    )

    df = df.round(2)
    df.to_csv(f"csv/SS-datasets-{metric}.csv")

    df.columns.name = None
    df.index.names = ["c", "Model"]
    df.columns = [
        col.replace(" CC", "")
        .replace(" SS", "")
        .replace("[IMG] ", "")
        .replace("[TAB] ", "")
        .replace("[TXT] ", "")
        for col in df.columns
    ]
    if metric in ["accuracy"]:
        display(df)

    os.makedirs(DIR, exist_ok=True)
    for i, df_half in enumerate(
        [
            df.iloc[:, : len(df.columns) // 2],
            df.iloc[:, len(df.columns) // 2 :],
        ]
    ):
        latex_table = df_half.style.format(precision=2).to_latex(
            column_format="l|c|" + "c" * len(df_half.columns)
        )
        latex_table = merge_latex_headers(latex_table, scaling=0.75)
        with open(os.path.join(DIR, f"SS-{metric}-p{i+1}.tex"), "w") as f:
            f.write(latex_table)

# %%
results_df[
    (results_df.dataset.str.contains("CC"))
    & (results_df.model == "nnPUss")
    & (results_df.label_frequency == 0.1)
]

# %%
sizes = {
    "[IMG] Beans SS": 1034,
    "[IMG] CIFAR SS": 60000,
    "[IMG] Chest X-ray SS": 4077,
    "[IMG] DogFood SS": 2500,
    "[IMG] EuroSAT SS": 21600,
    "[IMG] FashionMNIST SS": 60000,
    "[IMG] MNIST SS": 60000,
    "[IMG] Oxford Pets SS": 7390,
    "[IMG] Snacks SS": 4838,
    "[TAB] California SS": 20634,
    "[TAB] Credit SS": 16714,
    "[TAB] Electricity SS": 38474,
    "[TAB] Wine SS": 2554,
    "[TXT] 20News SS": 11300,
    "[TXT] HateSpeech SS": 10900,
    "[TXT] IMDB SS": 25000,
    "[TXT] PoemSentiment SS": 892,
    "[TXT] SMSSpam SS": 5574,
}
