import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import Dataset
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
import numpy as np
import os

# Initialize the processor and model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained("path_to_your_trained_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_image(image):
    """Converts and preprocesses an image for the LayoutLMv3 model."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("Unsupported image type")

    encoded_inputs = processor(image, padding="max_length", truncation=True, return_tensors="pt")
    return encoded_inputs

def predict_and_extract_details(encoded_inputs):
    """Uses the model to predict and extract details from the encoded inputs."""
    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.cpu().numpy()

def pdf_to_images(pdf_file):
    """Converts a PDF file to a list of images."""
    return convert_from_path(pdf_file)

def extract_details_from_pdf(pdf_file):
    """Extracts details from a PDF file and returns them in a dictionary."""
    images = pdf_to_images(pdf_file)
    details = []

    for image in images:
        encoded_inputs = preprocess_image(image)
        predictions = predict_and_extract_details(encoded_inputs)
        
        # Extract and format predictions (you may need to adapt this part depending on your labels and output format)
        for i, prediction in enumerate(predictions[0]):
            detail = {
                "Image": i + 1,
                "Prediction": prediction,
                # Add more fields based on your model's output and requirements
            }
            details.append(detail)

    return details

def details_to_excel(details, output_file="output.xlsx"):
    """Saves the extracted details to an Excel file."""
    df = pd.DataFrame(details)
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    pdf_file = "input.pdf"  # Replace with your input PDF file
    details = extract_details_from_pdf(pdf_file)
    details_to_excel(details, "extracted_details.xlsx")
    print("Details extracted and saved to 'extracted_details.xlsx'")
