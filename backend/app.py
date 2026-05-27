from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Multi AI Model Serving API",
    description="FastAPI backend serving multiple Hugging Face models",
    version="2.0.0"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOAD MODELS
# =========================================================

print("Loading models...")

# 1. Sentiment Analysis
sentiment_classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# 2. Summarization
summarizer = pipeline(
    task="summarization",
    model="Falconsai/text_summarization"
)

# 3. Named Entity Recognition
ner_pipeline = pipeline(
    task="ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# 4. Question Answering
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# 5. Translation (English to French)
translator = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

# 6. Text Generation
text_generator = pipeline(
    task="text-generation",
    model="distilgpt2"
)


print("All models loaded successfully!")

# =========================================================
# REQUEST SCHEMAS
# =========================================================

class TextInput(BaseModel):
    text: str

class QAInput(BaseModel):
    question: str
    context: str



# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Multi AI Model Serving API is running!"
    }

# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

# =========================================================
# MODELS
# =========================================================

@app.get("/models")
def get_models():
    return {
        "available_models": [
            "sentiment-analysis",
            "summarization",
            "named-entity-recognition",
            "question-answering",
            "translation",
            "text-generation",
        ]
    }

# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

@app.post("/sentiment")
def analyze_sentiment(data: TextInput):

    result = sentiment_classifier(data.text)

    return {
        "task": "sentiment-analysis",
        "input": data.text,
        "prediction": result
    }

# =========================================================
# SUMMARIZATION
# =========================================================

@app.post("/summarize")
def summarize_text(data: TextInput):

    result = summarizer(
        data.text,
        max_length=60,
        min_length=20,
        do_sample=False
    )

    return {
        "task": "summarization",
        "input": data.text,
        "summary": result
    }

# =========================================================
# NAMED ENTITY RECOGNITION
# =========================================================

@app.post("/ner")
def extract_entities(data: TextInput):

    result = ner_pipeline(data.text)

    formatted_entities = []

    for entity in result:

        formatted_entities.append({

            "entity_group": entity["entity_group"],

            "word": entity["word"],

            "score": float(entity["score"])
        })

    return {
        "entities": formatted_entities
    }

# =========================================================
# QUESTION ANSWERING
# =========================================================

@app.post("/qa")
def answer_question(data: QAInput):

    result = qa_pipeline(
        question=data.question,
        context=data.context
    )

    return {
        "task": "question-answering",
        "question": data.question,
        "context": data.context,
        "answer": result
    }

# =========================================================
# TRANSLATION
# =========================================================

@app.post("/translate")
def translate_text(data: TextInput):

    result = translator(data.text)

    return {
        "task": "translation",
        "input": data.text,
        "translation": result
    }

# =========================================================
# TEXT GENERATION
# =========================================================

@app.post("/generate")
def generate_text(data: TextInput):

    result = text_generator(
        data.text,
        max_length=50,
        num_return_sequences=1
    )

    return {
        "task": "text-generation",
        "input": data.text,
        "generated_text": result
    }
