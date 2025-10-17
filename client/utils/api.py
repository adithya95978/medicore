import requests
from config import API_URL


def upload_pdfs_api(files):
    files_payload=[ ("files",(f.name,f.read(),"application/pdf")) for f in files]
    return requests.post(f"{API_URL}/upload_pdfs/",files=files_payload)

def ask_question(question):
    return requests.post(f"{API_URL}/ask/",data={"question":question})

def get_answer_with_image(question: str, image_file):
    """Sends a question and an image to the backend."""
    files_payload = {'image': (image_file.name, image_file.getvalue(), image_file.type)}
    data_payload = {'question': question}
    return requests.post(f"{API_URL}/ask_with_image", files=files_payload, data=data_payload)
