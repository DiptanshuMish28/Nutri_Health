import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re

# Path to Tesseract executable (if required)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Step 1: Preprocess the image to improve OCR results
def preprocess_image(image_path):
    try:
        # Open the image file
        img = Image.open(image_path)

        # Convert to grayscale
        img = img.convert('L')

        # Apply a filter to improve sharpness
        img = img.filter(ImageFilter.SHARPEN)

        # Apply thresholding to binarize the image
        img = img.point(lambda x: 0 if x < 140 else 255)

        # Optionally save preprocessed image for inspection
        img.save('preprocessed_image.png')

        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Step 2: Extract text from the preprocessed image
def extract_text_from_image(preprocessed_image):
    try:
        # Use pytesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(preprocessed_image)

        return extracted_text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

# Step 3: Extract medical fields from the text using regular expressions
def extract_medical_fields(extracted_text):
    patterns = {
        "Age": r"Age/Gender\s*:\s*(\d+)",
        "Total_Bilirubin": r"TOTAL BILIRUBIN\s*([\d.]+)",
        "Direct_Bilirubin": r"DIRECT BILIRUBIN\s*([\d.]+)",
        "Alkaline_Phosphotase": r"ALKALINE PHOSPHATASE\s*([\d.]+)",
        "Alamine_Aminotransferase": r"SGPT\s*([\d.]+)",
        "Aspartate_Aminotransferase": r"SGOT\s*([\d.]+)",
        "Total_Protiens": r"TOTAL PROTEINS\s*([\d.]+)",
        "Albumin": r"ALBUMIN\s*([\d.]+)",
        "Albumin_and_Globulin_Ratio": r"A/G RATIO\s*([\d.]+)"
    }

    extracted_values = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, extracted_text)
        extracted_values[field] = match.group(1) if match else "Not found"

    return extracted_values

# Main execution
if __name__ == "__main__":
    # Step 1: Preprocess the image
    image_path = "lft.jpg"
    preprocessed_image = preprocess_image(image_path)

    # Step 2: Perform OCR on the preprocessed image
    if preprocessed_image:
        extracted_text = extract_text_from_image(preprocessed_image)

        # Step 3: Extract medical fields if OCR was successful
        if extracted_text:
            medical_fields = extract_medical_fields(extracted_text)

            print("Extracted Medical Information:")
            for field, value in medical_fields.items():
                print(f"{field}: {value}")

            # Optionally save the extracted fields to a file
            with open("extracted_medical_data.txt", "w") as text_file:
                for field, value in medical_fields.items():
                    text_file.write(f"{field}: {value}\n")      