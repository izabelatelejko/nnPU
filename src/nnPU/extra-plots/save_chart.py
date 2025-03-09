import os

import altair as alt
import cairosvg


def save_chart(chart: alt.Chart, DIR: str, img_name: str):
    os.makedirs(DIR, exist_ok=True)
    chart.save(
        os.path.join(
            DIR,
            f"{img_name}.png",
        ),
        scale_factor=3,
    )
    chart.save(
        os.path.join(
            DIR,
            f"{img_name}.svg",
        ),
    )
    cairosvg.svg2pdf(
        url=os.path.join(
            DIR,
            f"{img_name}.svg",
        ),
        write_to=os.path.join(
            DIR,
            f"{img_name}.pdf",
        ),
    )
