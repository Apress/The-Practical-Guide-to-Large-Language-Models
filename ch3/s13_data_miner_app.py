from pydantic import BaseModel
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines


# Patient Schema
class Patient(BaseModel):
    patient_name: str
    gender: str
    age: int
    symptoms: list[str]
    diagnosis: str
    treatment_plan: list[str]


# Load the model and JSON generator
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map = "auto"
    ),
    AutoTokenizer.from_pretrained(MODEL_ID),
)


# Inference logic to populate each field
def extract_fields(medical_note):
    prompt = "Read the following medical note and extract structured JSON according to the schema: " + medical_note
    try:
        raw = model(prompt, Patient, max_new_tokens = 256)
        print(f"Raw output: {raw}")
        patient = Patient.model_validate_json(raw)
        return (
            patient.patient_name,
            patient.gender,
            patient.age,
            "\n".join(patient.symptoms),
            patient.diagnosis,
            "\n".join(patient.treatment_plan),
        )
    except Exception as e:
        return ("Error", "unknown", 0, "Error", str(e), "")


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Medical Note Data Miner")

    with gr.Row():
        medical_note = gr.Textbox(
            label = "Medical Note",
            placeholder = "Paste the medical case here...",
            lines = 5
        )

    with gr.Row():
        submit_btn = gr.Button("Extract Information")

    gr.Markdown("### Extracted Information")

    with gr.Row():
        patient_name = gr.Textbox(label = "Patient Name")
        gender = gr.Textbox(label = "Gender")
        age = gr.Number(label = "Age", precision = 0)

    with gr.Row():
        symptoms = gr.Textbox(label = "Symptoms (one per line)", lines = 5)
        diagnosis = gr.Textbox(label = "Diagnosis")
        treatment_plan = gr.Textbox(label = "Treatment Plan (one per line)", lines = 5)

    submit_btn.click(
        extract_fields,
        inputs = [medical_note],
        outputs = [
            patient_name,
            gender,
            age,
            symptoms,
            diagnosis,
            treatment_plan
        ]
    )

demo.launch(share = True)
