# RAG-Powered Insurance Chatbot

## Introduction

Welcome to the RAG-Powered Insurance Chatbot repository. This project demonstrates the use of Retrieval-Augmented Generation (RAG) to create a chatbot capable of answering questions based on the content of an insurance policy document. The project leverages state-of-the-art models from the Hugging Face Transformers library to encode questions and contexts, and generate accurate and contextually relevant answers.

## Live Demo

Check out the live demo of the chatbot: [RAG-Powered Insurance Chatbot](https://rag-powered-insurance-chatbot.streamlit.app/)

## Features

- **Question Answering**: Ask questions about the insurance policy and get accurate answers.
- **User-Friendly Interface**: Simple and intuitive interface built with Streamlit.
- **Model Integration**: Utilizes Dense Passage Retrieval (DPR) and BART models from Hugging Face.

## Dataset Construction

The dataset was constructed by extracting query-response pairs from the [Churchill Motor Insurance Policy Booklet](https://assets.churchill.com/motor-docs/policy-booklet-0923.pdf). A total of 30 diverse query-response pairs were selected to ensure comprehensive coverage of the document’s content.

## Model Details

### Question and Context Encoding

- **DPR (Dense Passage Retrieval)**: Used for encoding questions and contexts.
  - **Question Encoder**: `facebook/dpr-question_encoder-single-nq-base`
  - **Context Encoder**: `facebook/dpr-ctx_encoder-single-nq-base`

### Answer Generation

- **BART (Bidirectional and Auto-Regressive Transformers)**: Used for generating answers.
  - **Model**: `facebook/bart-large-cnn`

## Evaluation Metrics

### BLEU Score

The BLEU (Bilingual Evaluation Understudy) score was used to evaluate the accuracy of the generated answers. This metric measures the similarity between the generated answer and the reference answer based on n-gram overlaps.

- **Average BLEU Score**: 0.268
- **Interpretation**: An average BLEU score of 0.268 indicates that, on average, there is a 26.8% overlap between the generated answers and the reference answers. This suggests that the model captures a reasonable amount of relevant information from the context but still has room for improvement to generate more precise and contextually accurate answers. Higher BLEU scores generally indicate better performance in generating answers that closely match the reference answers.

  ## Accuracy Improvements

To improve the model's accuracy, the following techniques were applied:

1. **Data Augmentation**: Increasing the diversity and number of training examples.
2. **Fine-Tuning**: Adjusting the models with the provided dataset to better capture the context.
3. **Hyperparameter Optimization**: Tuning parameters to find the best configuration for optimal performance.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/bechosen-spec/rag-powered-insurance-chatbot.git
cd rag-powered-insurance-chatbot
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Download Models**

The necessary models will be downloaded automatically when the script is run for the first time.

### Running the App

Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main application script for Streamlit.
- `data/`: Directory containing the dataset (if applicable).
- `models/`: Directory where models are saved (if applicable).
- `README.md`: Project documentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Contact

For any questions or inquiries, please contact Emmanuel at bonifacechosen100@gmail.com.

