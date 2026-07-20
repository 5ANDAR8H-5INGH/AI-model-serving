from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai 
from pydantic import BaseModel
from transformers import pipeline

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows requests from specific domains
    allow_credentials=True,           # Allows cookies to be included in requests
    allow_methods=["*"],              # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all headers
)

class PromptRequest(BaseModel):
    prompt : str

classifier = pipeline("summarization")
classifier_1 = pipeline("sentiment-analysis")
classifier_2 = pipeline("text-generation",model="gpt2")

@app.get('/')
def homepage():
    return {"message":"Welcome to HomePage"}

@app.post('/generate_text')
def generate_text(query: PromptRequest):
    result = classifier_2(
        query.prompt,
        max_length=100,
        num_return_sequences=1
    )
    generated_text = result[0]['generated_text']
    return {
        "Response": generated_text
    }

@app.post('/summarize')
def summarize(query: PromptRequest):
    result = classifier(
        query.prompt,
        max_length=80,
        min_length=20,
        do_sample=False
    )
    summary = result[0]['summary_text']
    return {
        "Summary": summary
    }

@app.post('/sentiment_analysis')
def sentiment_analysis(query: PromptRequest):
    result = classifier_1(query.prompt)
    return {
        "Sentiment": result[0]['label'],
        "Score": round(result[0]['score'] * 100, 2)
    }

@app.post('/qna')
def qna(query:PromptRequest):
    content = gemini_model.generate_content(query.prompt)
    return {"Response": content.text}
