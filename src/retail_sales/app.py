import gradio as gr

from . import model

CATEGORIES = ["Beauty", "Clothing", "Electronics"]


def build_interface() -> gr.Interface:
    pipeline = model.load()

    def predict(age, gender, category, month, day):
        amount = model.predict_one(pipeline, age, gender, category, month, day)
        return f"Predicted Transaction Amount: ${amount:.2f}"

    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Slider(18, 100, value=30, step=1, label="Customer Age"),
            gr.Dropdown(["Male", "Female"], value="Female", label="Gender"),
            gr.Dropdown(CATEGORIES, value="Beauty", label="Product Category"),
            gr.Slider(1, 12, value=6, step=1, label="Month of Purchase"),
            gr.Slider(1, 31, value=15, step=1, label="Day of Month"),
        ],
        outputs=gr.Textbox(label="Prediction Result"),
        title="Retail Sales Predictor",
        description="Predict the expected Total Amount of a transaction.",
    )


if __name__ == "__main__":
    build_interface().launch()
