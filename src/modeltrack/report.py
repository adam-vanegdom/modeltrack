import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime


def plot_loss(path, current_epoch, train_loss, test_loss):
    plotname = os.path.join(path, "training_loss_curve.png")
    fig = plt.figure()
    plt.axes().set_facecolor("#fbc9bc")
    plt.plot(
        range(1, current_epoch + 1), train_loss, color="#ff6050", label="Training Loss"
    )
    plt.plot(range(1, current_epoch + 1), test_loss, color="#19214e", label="Test Loss")
    plt.xlabel("Epoch Count")
    plt.ylabel("Model Loss")
    plt.legend()
    fig.savefig(plotname, bbox_inches="tight")
    plt.close()


def produce_summary_pdf(model_name, img_path, hyperparams, model_arch, train_stats):
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

    # Model Configuration Section
    pdf.cell(150, 10, "Model Configuration:", 0, 2)
    pdf.cell(30, 10, "Parameter", 1, 0)
    pdf.cell(140, 10, "Value", 1, 2)
    pdf.set_text_color(255, 96, 80)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(-30)
    attributes = ["model_dir", "log_dir", "current_epoch", "overwrite", "exp_name"]
    for i, val in enumerate(hyperparams):
        if val not in attributes:
            pdf.cell(30, 10, "%s" % (val), 1, 0)
            pdf.cell(140, 10, "%s" % (hyperparams[val]), 1, 2)
            pdf.cell(-30)
    pdf.cell(90, 3, "", 0, 2)

    # Model Performance Section
    pdf.set_text_color(25, 33, 78)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(150, 10, "Model Performance Stats:", 0, 2)
    pdf.set_font("Helvetica", "", 12)

    loss = train_stats["test_loss"]
    acc = train_stats["test_acc"]

    pdf.set_text_color(255, 96, 80)
    pdf.cell(35, 6, "Best Loss:", 0, 0)
    pdf.cell(45, 6, "{:.3f} (Epoch {})".format(min(loss), loss.index(min(loss))), 0, 0)
    pdf.cell(60, 6, "Training Duration:", 0, 0)
    pdf.cell(30, 6, "{:.3f} (s)".format(train_stats["total_dur"]), 0, 2)
    pdf.cell(-140)
    pdf.cell(35, 6, f"Best Accuracy:", 0, 0)
    pdf.cell(45, 6, "{:.3f} (Epoch {})".format(min(acc), acc.index(min(acc))), 0, 0)
    pdf.cell(60, 6, "Average Epoch Duration:", 0, 0)
    pdf.cell(30, 6, "{:.3f} (s)".format(train_stats["total_dur"]), 0, 2)
    pdf.cell(-140)
    pdf.cell(90, 3, "", 0, 2)

    # Loss Curve Section
    pdf.set_text_color(25, 33, 78)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(150, 10, "Model Loss Curve:", 0, 2)
    pdf.image(img_path, x=None, y=None, w=160, h=0, type="PNG", link="")

    # Second Page of Report
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.cell(20, 20)

    # Model Arch Section
    pdf.cell(150, 20, "Model Configuration:", 0, 2)
    pdf.set_font("Helvetica", "", 12)
    if model_arch is None:
        model_arch = "No model configuration was provided"
    pdf.set_text_color(255, 96, 80)
    pdf.multi_cell(180, 8, str(model_arch))

    # Third Page of Report
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.cell(20, 20, " ")

    # Training Loss Section
    pdf.set_text_color(25, 33, 78)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(150, 20, "Detailed Loss Output:", 0, 2)
    pdf.cell(40, 8, "Epoch", 1, 0, "C")
    pdf.cell(30, 8, "Train Loss", 1, 0, "C")
    pdf.cell(30, 8, "Test Loss", 1, 0, "C")
    pdf.cell(30, 8, "Train Acc", 1, 0, "C")
    pdf.cell(30, 8, "Test Acc", 1, 2, "C")
    pdf.set_text_color(255, 96, 80)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(-130)
    for i in range(0, len(train_stats["train_loss"])):
        pdf.cell(40, 8, "{}".format((i + 1)), 1, 0, "C")
        pdf.cell(30, 8, "{:.3f}".format((train_stats["train_loss"][i])), 1, 0, "C")
        pdf.cell(30, 8, "{:.3f}".format((train_stats["test_loss"][i])), 1, 0, "C")
        pdf.cell(30, 8, "{:.3f}".format((train_stats["train_acc"][i])), 1, 0, "C")
        pdf.cell(30, 8, "{:.3f}".format((train_stats["test_acc"][i])), 1, 2, "C")
        pdf.cell(-130)
    pdf.cell(90, 3, "", 0, 2)

    pdf.output(os.path.join(os.path.dirname(img_path), "training_summary.pdf"), "F")
