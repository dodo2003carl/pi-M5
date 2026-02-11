import sys
import os

pdf_path = "Presentación del PI M5.pdf"

try:
    import pypdf
    print("Using pypdf")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("PDF content written to pdf_content.txt")
except ImportError:
    try:
        import PyPDF2
        print("Using PyPDF2")
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(text)
    except ImportError:
        print("No PDF library found. Please install pypdf or PyPDF2.")
