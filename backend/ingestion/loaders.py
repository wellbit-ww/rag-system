import fitz

from llama_index.core.schema import Document


def load_pdf(file_path: str):

    pdf = fitz.open(file_path)

    full_text = ""

    for page in pdf:

        text = page.get_text()

        full_text += text + "\n"

    pdf.close()

    return [
        Document(text=full_text)
    ]