from flask import Flask, render_template, request , jsonify
import os
import openai
import json
from PyPDF2 import PdfReader
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import time
import pdfplumber
from langchain.llms import OpenAI
from PyPDF2 import PdfFileReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback





# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
OPENAI_API_KEY = 'sk-bn9QJOoRDMmNeRE4DcapT3BlbkFJZsUHNWNInibxg298i7NYT'

app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'paper'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


# Chunk from local
chunks_file = "chunks.json"


# Save Chunk data from the folder from the path provided
def save_chunks(chunks):
    with open(chunks_file, "w") as file:
        json.dump(chunks, file)


def load_chunks():
    try:
        with open(chunks_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def read_files(uploaded_files: list):
    all_text = ""
    for folder_path in uploaded_files:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                    all_text += file.read() + "\n"
            elif filename.endswith(".pdf"):
                pdf_file_path = os.path.join(folder_path, filename)
                with pdfplumber.open(pdf_file_path) as pdf:
                    for page in pdf.pages:
                        all_text += page.extract_text() + "\n"  # No need for an encoding argument

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Create chunk datas
        chunks = text_splitter.split_text(all_text)
        # Save the chunks into a file
        save_chunks(chunks)



def is_question_valid(question: str, text: str) -> bool:

    # Check if the question is not too general (e.g., longer than a certain length)
    if len(question) < 20:  # You can adjust the length threshold
        return True

    return False


def split_text_into_chunks(text, max_tokens=100):
    # Split the text into chunks that fit within the token limit
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) < max_tokens:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

conversation = []

@app.route("/", methods=["GET", "POST"])

def index():

    folder_paths = ["./paper", "./pdf", "./resume"]
    read_files(folder_paths)

    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        return render_template("index.html", conversation=conversation, form=form)

    if request.method == "POST":
        chunks = load_chunks()
        question = request.form["question"]
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        if question:
            openai.api_key =os.getenv(OPENAI_API_KEY)
            docs = knowledge_base.similarity_search(question)
            llm = OpenAI(openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=question)
        conversation.append({"user": question, "assistant": response})

    return render_template("index.html", conversation=conversation, form=form)


if __name__ == "__main__":
    app.run(debug=True)