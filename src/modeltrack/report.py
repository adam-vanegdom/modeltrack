import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime


def plot_loss(path, current_epoch, train_loss, test_loss):
    plotname = os.path.join(path, "training_loss_curve.png")
    fig = plt.figure()
    plt.plot(range(1, current_epoch + 1), train_loss, color="r", label="Training Loss")
    plt.plot(range(1, current_epoch + 1), test_loss, color="b", label="Test Loss")
    plt.xlabel("Epoch Count")
    plt.ylabel("Model Loss")
    plt.legend()
    fig.savefig(plotname, bbox_inches="tight")
    plt.close()


def produce_summary_pdf(model_name, img_path, hyperparams, model_arch):
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 10)
    pdf.set_font("Helvetica", "BI", 16)
    pdf.set_text_color(25, 33, 78)
    pdf.set_draw_color(25, 33, 78)
    pdf.cell(20)
    pdf.cell(
        200,
        10,
        "Model Training Summary: {}".format(model_name.upper()),
        0,
        2,
    )
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        200,
        5,
        dt_string,
        0,
        2,
    )
    pdf.cell(150, 10, "Model Configuration", 0, 2)
    pdf.cell(50, 10, "Parameter", 1, 0)
    pdf.cell(130, 10, "Value", 1, 2)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(-50)
    attributes = [
        "exp_name",
        "description",
        "batch_size",
        "max_epochs",
        "learning_rate",
    ]
    for i, val in enumerate(attributes):
        pdf.cell(50, 10, "%s" % (val), 1, 0)
        pdf.cell(130, 10, "%s" % (hyperparams[val]), 1, 2)
        pdf.cell(-50)
    pdf.cell(90, 10, " ", 0, 2)
    # pdf.cell(-30)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(150, 10, "Model Loss Curve:", 0, 2)
    pdf.image(img_path, x=None, y=None, w=160, h=0, type="PNG", link="")

    # Second Page of Report
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.cell(20, 20)
    pdf.cell(150, 20, "Model Configuration:", 0, 2)
    pdf.set_font("Helvetica", "", 12)

    if model_arch is None:
        model_arch = "No model configuration was provided"

    pdf.multi_cell(180, 8, str(model_arch))
    pdf.output(os.path.join(os.path.dirname(img_path), "training_summary.pdf"), "F")
