from transformers import pipeline
import gradio as gr
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

model = pipeline("summarization", model="Falconsai/text_summarization")

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

# with gr.Blocks() as demo:
textbox = gr.Textbox(placeholder="Enter text block to summarize", lines=4)
interface = gr.Interface(fn=predict, inputs=textbox, outputs="text")

interface.launch()