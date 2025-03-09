# %%
import altair as alt
import numpy as np
import pandas as pd
import scipy.stats
from save_chart import save_chart

domain = np.linspace(-5, 5, 100)
positive_dist = scipy.stats.distributions.norm.pdf(domain, loc=2)
negative_dist = scipy.stats.distributions.norm.pdf(domain, loc=-2)

pi = 0.5
true_distribution = 0.5 * positive_dist + 0.5 * negative_dist

dfs = []
for c in [0.1, 0.5, 0.9]:
    A = 1 / (1 - c + c * pi)

    # positive_ss_density = c * pi * positive_dist
    # unlabeled_ss_density = (1 - c) * pi * positive_dist + (1 - pi) * negative_dist
    # ss_dataset = positive_ss_density + unlabeled_ss_density
    positive_ss_density = positive_dist
    unlabeled_ss_density = (1 - c) * pi * positive_dist + (1 - pi) * negative_dist / (
        (1 - c) * pi + (1 - pi)
    )

    # positive_cc_density = A * c * pi * positive_dist
    # unlabeled_cc_density = A * (1 - c) * true_distribution
    # cc_dataset = positive_cc_density + unlabeled_cc_density
    positive_cc_density = positive_dist
    unlabeled_cc_density = true_distribution

    true_data = pd.DataFrame(
        {
            "Value": domain,
            "Positive / Labeled density": positive_dist,
            "Negative density": negative_dist,
            # "Dataset density": true_distribution,
            "Label frequency": f"c = {c}",
        }
    ).melt(["Value", "Label frequency"], var_name="Density type", value_name="Density")
    true_data["Dataset type"] = "True distribution"

    ss_data = pd.DataFrame(
        {
            "Value": domain,
            "Positive / Labeled density": positive_ss_density,
            "Unlabeled density": unlabeled_ss_density,
            # "Dataset density": ss_dataset,
            "Label frequency": f"c = {c}",
        }
    ).melt(["Value", "Label frequency"], var_name="Density type", value_name="Density")
    ss_data["Dataset type"] = "Single sample"

    cc_data = pd.DataFrame(
        {
            "Value": domain,
            "Positive / Labeled density": positive_cc_density,
            "Unlabeled density": unlabeled_cc_density,
            # "Dataset density": cc_dataset,
            "Label frequency": f"c = {c}",
        }
    ).melt(["Value", "Label frequency"], var_name="Density type", value_name="Density")
    cc_data["Dataset type"] = "Case control"

    dfs.append(true_data)
    dfs.append(ss_data)
    dfs.append(cc_data)

colors = [
    # "gray",
    "#1F77B4",
    "#D62728",
    "#9467BD",
]


# alt.themes.enable("googlecharts")
alt.themes.enable("default")

chart = (
    alt.Chart(pd.concat(dfs), width=250, height=60)
    .mark_line(opacity=0.8)
    .encode(
        alt.X("Value"),
        alt.Y("Density"),
        alt.Color(
            "Density type",
            sort=[
                # "Dataset density",
                "Positive / Labeled density",
                "Negative density",
                "Unlabeled density",
            ],
        ).legend(title=None, labelLimit=400),
        # .legend(orient="bottom", title=None, labelLimit=400),
    )
    .facet(
        row="Label frequency:N",
        column=alt.Column(
            "Dataset type", sort=["True distribution", "Single sample", "Case control"]
        ),
    )
    .configure_range(category=alt.RangeScheme(colors))
    # .properties(title="Density")
    # .configure_title(fontSize=20, anchor="middle")
    .configure_axis(labelFontSize=13, titleFontSize=13)
    .configure_header(titleFontSize=15, labelFontSize=15)
    .configure_legend(labelFontSize=13, titleFontSize=13)
)

save_chart(chart, "img", "ss_vs_cc")
chart

# %%
