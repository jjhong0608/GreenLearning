import plotly
from pathlib import Path

if __name__ == '__main__':
    # file = Path("/home/jjhong0608/Documents/GreenNetResearch/Coupler/checkpoints/For_Workshop/CouplingNetResults/Box_Plots/box_plots.json")
    # fig = plotly.io.read_json(str(file))
    # fig.layout.title= "Relative L2-Error Distribution"
    # fig.layout.showlegend = False
    # fig.layout.yaxis.title = "Relative L2-Error"
    # fig.layout.yaxis.tickformat = "%.0f"
    # fig.layout.xaxis.tickangle = -20
    # fig.write_image(str(file.with_name(file.stem + "_modify").with_suffix(".pdf")))
    # print(fig)
    # print(fig.layout.xaxis)

    root = Path("/home/jjhong0608/Documents/GreenNetResearch/Coupler/checkpoints/For_Workshop/CouplingNetResults")
    dirs = ["Poisson", "Diffusion", "Diffusion_Reaction", "Convection_Diffusion_Reaction"]
    for dir in dirs:
        prob_dir = root / dir
        print(f"prob_dir: {prob_dir}")
        for file in prob_dir.glob("*.json"):
            name = file.stem
            fig = plotly.io.read_json(file)
            title = fig.layout.title.text
            if title == "Exact solution u":
                fig.update_layout(title="Reference Solution")
            elif title == "Predicted solution u_pred":
                fig.update_layout(title="Predicted Solution")
            elif title == "Source f":
                fig.update_layout(title="Source")
            else:
                print(f"Unknown title: {title}")
                raise NotImplementedError
            new_name = name + "_modify"
            fig.write_image(str(file.with_name(new_name).with_suffix(".pdf")))
            print(file)
            print(file.with_name(new_name).with_suffix(".pdf"))
