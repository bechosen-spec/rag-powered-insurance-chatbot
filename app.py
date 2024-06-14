# import streamlit as st
# from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRContextEncoder
# from transformers import BartForConditionalGeneration, BartTokenizer
# import torch

# # Load the DPR question and context encoders
# question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# context_encoder_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# # Load the fine-tuned BART model for generation
# generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# generator_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# # Example Q&A pairs for context encoding
# qa_pairs = [
#     {"question": "What is covered under Section 1: Liability?", "answer": "Section 1 covers liability for injuries to other people and damage to their property."},
#     {"question": "How can I make a claim?", "answer": "To make a claim, call 0345 878 6261 or 0800 328 9150 for windscreen claims."},
#     {"question": "What does DriveSure provide?", "answer": "DriveSure is a telematics insurance product that monitors driving style to adjust premiums."},
#     {"question": "What are the exclusions for accidental damage?", "answer": "Exclusions include damage caused by someone convicted of driving under the influence of drink or drugs."},
#     {"question": "How is personal accident covered?", "answer": "Personal accident covers injuries or death while traveling in or getting in or out of the car, up to specified amounts."},
#     {"question": "What happens if my car is stolen?", "answer": "If your car is stolen, we can repair, replace, or repay you based on the market value of the car."},
#     {"question": "How do you define 'approved repairer'?", "answer": "An approved repairer is part of our network of contracted repairers authorized to carry out repairs after a claim."},
#     {"question": "What does the 'Vandalism Promise' cover?", "answer": "The Vandalism Promise ensures that no claim discount is unaffected if your car is damaged due to vandalism."},
#     {"question": "Is my electric car battery covered?", "answer": "Yes, the battery is covered if it is damaged as a result of an insured incident."},
#     {"question": "What is the excess for windscreen repairs?", "answer": "The excess for windscreen repairs is specified in your car insurance details."},
#     {"question": "What is included in 'Personal belongings' coverage?", "answer": "It covers personal belongings lost or damaged by fire, theft, or accident while in the car, up to specified amounts."},
#     {"question": "How do you handle car keys theft?", "answer": "We replace stolen car keys and locks, including locksmith charges, after verifying with a police crime reference number."},
#     {"question": "What is the coverage for driving other cars?", "answer": "Coverage for driving other cars is limited to third-party liability as shown in your certificate of motor insurance."},
#     {"question": "What is the limit for property damage liability?", "answer": "The limit for property damage liability is £20,000,000 per accident."},
#     {"question": "What happens if my car is written off?", "answer": "If your car is written off, the policy will settle based on market value, and all cover will end unless agreed otherwise."},
#     {"question": "What is 'Motor Legal Cover'?", "answer": "Motor Legal Cover provides up to £100,000 for legal costs in case of a road traffic accident or motoring offence."},
#     {"question": "How does the 'New car replacement' benefit work?", "answer": "If your new car is stolen or written off within the first year, it will be replaced with one of the same make and model."},
#     {"question": "How are 'Hotel expenses' covered?", "answer": "Hotel expenses are covered if you cannot drive your car after an accident, up to specified amounts."},
#     {"question": "What are the 'Territorial limits'?", "answer": "The territorial limits include Great Britain, Northern Ireland, the Channel Islands, and the Isle of Man."},
#     {"question": "How does the 'Guaranteed Hire Car Plus' work?", "answer": "Guaranteed Hire Car Plus provides a hire car of a similar size while your car is being repaired, written off, or stolen."},
#     {"question": "What is the process for 'Making a claim'?", "answer": "The process involves providing personal details, policy number, car registration, and a description of the loss or damage."},
#     {"question": "What does 'Comprehensive Plus' cover?", "answer": "Comprehensive Plus includes all the benefits of Comprehensive cover with additional perks like Guaranteed Hire Car Plus."},
#     {"question": "What are the conditions for using a courtesy car?", "answer": "A courtesy car is provided while your car is being repaired by an approved repairer, subject to availability and certain conditions."},
#     {"question": "What is the coverage for 'Child car seats'?", "answer": "Child car seats are covered for replacement if damaged by fire, theft, or accident, even without visible damage."},
#     {"question": "How are 'Medical expenses' handled?", "answer": "Medical expenses are covered if people are injured in an accident involving your car, up to specified amounts."},
#     {"question": "What does 'Accidental damage' cover?", "answer": "Accidental damage covers damage to your car that happens by accident, with options to repair, replace, or repay."},
#     {"question": "How are 'In-car entertainment systems' covered?", "answer": "In-car entertainment systems fitted when the car was made are covered for replacement if damaged by fire or theft."},
#     {"question": "What is the 'Uninsured Driver Promise'?", "answer": "If an uninsured driver hits your car, the no claim discount will not be affected and the excess will be refunded."},
#     {"question": "What are 'Excesses' and how are they applied?", "answer": "Excesses are amounts you pay towards a claim. They vary depending on the driver and are specified in your car insurance details."},
#     {"question": "What is the 'Motor legal helpline'?", "answer": "The motor legal helpline provides confidential legal advice on any private motoring legal problem, available 24/7."}
# ]

# # Encode all contexts (answers)
# contexts = [pair['answer'] for pair in qa_pairs]
# context_embeddings = []
# for context in contexts:
#     inputs = context_encoder_tokenizer(context, return_tensors='pt', truncation=True, padding=True)
#     context_embedding = context_encoder(**inputs).pooler_output
#     context_embeddings.append(context_embedding)

# # Concatenate all context embeddings
# context_embeddings = torch.cat(context_embeddings, dim=0)

# def generate_answer(question, context_embeddings):
#     # Encode the question
#     question_inputs = question_encoder_tokenizer(question, return_tensors='pt', truncation=True, padding=True)
#     question_embedding = question_encoder(**question_inputs).pooler_output
    
#     # Compute dot product similarity
#     similarities = torch.matmul(question_embedding, context_embeddings.T)
#     best_match_idx = torch.argmax(similarities, dim=1).item()
    
#     # Retrieve the best context
#     best_context = contexts[best_match_idx]
    
#     # Generate the answer using BART
#     generator_inputs = generator_tokenizer(question + " " + best_context, return_tensors='pt', truncation=True, padding=True)
#     summary_ids = generator_model.generate(generator_inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
#     answer = generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
#     return answer

# # Streamlit App
# st.title("RAG-Powered Insurance Chatbot")

# st.write("""
#     This chatbot provides answers based on an insurance policy document. 
#     Enter your question below and get an answer generated by our AI model.
# """)

# st.markdown("""
#     <style>
#     .main {
#         background-color: #f5f5f5;
#     }
#     .stTextInput input {
#         background-color: #ffffff;
#     }
#     .stButton button {
#         background-color: #4CAF50;
#         color: white;
#     }
#     </style>
# """, unsafe_allow_html=True)

# question = st.text_input("Enter your question:")

# if st.button("Get Answer"):
#     if question:
#         with st.spinner('Generating answer...'):
#             answer = generate_answer(question, context_embeddings)
#         st.write("**Answer:**", answer)
#     else:
#         st.write("Please enter a question.")


import os
import streamlit as st
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRContextEncoder
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Function to download models if not already available
def download_models():
    if not os.path.exists("dpr-question_encoder"):
        question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        question_encoder_tokenizer.save_pretrained("dpr-question_encoder")
        question_encoder.save_pretrained("dpr-question_encoder")
    
    if not os.path.exists("dpr-ctx_encoder"):
        context_encoder_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        context_encoder_tokenizer.save_pretrained("dpr-ctx_encoder")
        context_encoder.save_pretrained("dpr-ctx_encoder")
    
    if not os.path.exists("bart-large-cnn"):
        generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        generator_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        generator_tokenizer.save_pretrained("bart-large-cnn")
        generator_model.save_pretrained("bart-large-cnn")

# Download models if not already available
download_models()

# Load the models and tokenizers
question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("dpr-question_encoder")
question_encoder = DPRQuestionEncoder.from_pretrained("dpr-question_encoder")

context_encoder_tokenizer = DPRContextEncoderTokenizer.from_pretrained("dpr-ctx_encoder")
context_encoder = DPRContextEncoder.from_pretrained("dpr-ctx_encoder")

generator_tokenizer = BartTokenizer.from_pretrained("bart-large-cnn")
generator_model = BartForConditionalGeneration.from_pretrained("bart-large-cnn")

# Example Q&A pairs for context encoding
qa_pairs = [
    {"question": "What is covered under Section 1: Liability?", "answer": "Section 1 covers liability for injuries to other people and damage to their property."},
    {"question": "How can I make a claim?", "answer": "To make a claim, call 0345 878 6261 or 0800 328 9150 for windscreen claims."},
    {"question": "What does DriveSure provide?", "answer": "DriveSure is a telematics insurance product that monitors driving style to adjust premiums."},
    {"question": "What are the exclusions for accidental damage?", "answer": "Exclusions include damage caused by someone convicted of driving under the influence of drink or drugs."},
    {"question": "How is personal accident covered?", "answer": "Personal accident covers injuries or death while traveling in or getting in or out of the car, up to specified amounts."},
    {"question": "What happens if my car is stolen?", "answer": "If your car is stolen, we can repair, replace, or repay you based on the market value of the car."},
    {"question": "How do you define 'approved repairer'?", "answer": "An approved repairer is part of our network of contracted repairers authorized to carry out repairs after a claim."},
    {"question": "What does the 'Vandalism Promise' cover?", "answer": "The Vandalism Promise ensures that no claim discount is unaffected if your car is damaged due to vandalism."},
    {"question": "Is my electric car battery covered?", "answer": "Yes, the battery is covered if it is damaged as a result of an insured incident."},
    {"question": "What is the excess for windscreen repairs?", "answer": "The excess for windscreen repairs is specified in your car insurance details."},
    {"question": "What is included in 'Personal belongings' coverage?", "answer": "It covers personal belongings lost or damaged by fire, theft, or accident while in the car, up to specified amounts."},
    {"question": "How do you handle car keys theft?", "answer": "We replace stolen car keys and locks, including locksmith charges, after verifying with a police crime reference number."},
    {"question": "What is the coverage for driving other cars?", "answer": "Coverage for driving other cars is limited to third-party liability as shown in your certificate of motor insurance."},
    {"question": "What is the limit for property damage liability?", "answer": "The limit for property damage liability is £20,000,000 per accident."},
    {"question": "What happens if my car is written off?", "answer": "If your car is written off, the policy will settle based on market value, and all cover will end unless agreed otherwise."},
    {"question": "What is 'Motor Legal Cover'?", "answer": "Motor Legal Cover provides up to £100,000 for legal costs in case of a road traffic accident or motoring offence."},
    {"question": "How does the 'New car replacement' benefit work?", "answer": "If your new car is stolen or written off within the first year, it will be replaced with one of the same make and model."},
    {"question": "How are 'Hotel expenses' covered?", "answer": "Hotel expenses are covered if you cannot drive your car after an accident, up to specified amounts."},
    {"question": "What are the 'Territorial limits'?", "answer": "The territorial limits include Great Britain, Northern Ireland, the Channel Islands, and the Isle of Man."},
    {"question": "How does the 'Guaranteed Hire Car Plus' work?", "answer": "Guaranteed Hire Car Plus provides a hire car of a similar size while your car is being repaired, written off, or stolen."},
    {"question": "What is the process for 'Making a claim'?", "answer": "The process involves providing personal details, policy number, car registration, and a description of the loss or damage."},
    {"question": "What does 'Comprehensive Plus' cover?", "answer": "Comprehensive Plus includes all the benefits of Comprehensive cover with additional perks like Guaranteed Hire Car Plus."},
    {"question": "What are the conditions for using a courtesy car?", "answer": "A courtesy car is provided while your car is being repaired by an approved repairer, subject to availability and certain conditions."},
    {"question": "What is the coverage for 'Child car seats'?", "answer": "Child car seats are covered for replacement if damaged by fire, theft, or accident, even without visible damage."},
    {"question": "How are 'Medical expenses' handled?", "answer": "Medical expenses are covered if people are injured in an accident involving your car, up to specified amounts."},
    {"question": "What does 'Accidental damage' cover?", "answer": "Accidental damage covers damage to your car that happens by accident, with options to repair, replace, or repay."},
    {"question": "How are 'In-car entertainment systems' covered?", "answer": "In-car entertainment systems fitted when the car was made are covered for replacement if damaged by fire or theft."},
    {"question": "What is the 'Uninsured Driver Promise'?", "answer": "If an uninsured driver hits your car, the no claim discount will not be affected and the excess will be refunded."},
    {"question": "What are 'Excesses' and how are they applied?", "answer": "Excesses are amounts you pay towards a claim. They vary depending on the driver and are specified in your car insurance details."},
    {"question": "What is the 'Motor legal helpline'?", "answer": "The motor legal helpline provides confidential legal advice on any private motoring legal problem, available 24/7."}
]

# Encode all contexts (answers)
contexts = [pair['answer'] for pair in qa_pairs]
context_embeddings = []
for context in contexts:
    inputs = context_encoder_tokenizer(context, return_tensors='pt', truncation=True, padding=True)
    context_embedding = context_encoder(**inputs).pooler_output
    context_embeddings.append(context_embedding)

# Concatenate all context embeddings
context_embeddings = torch.cat(context_embeddings, dim=0)

def generate_answer(question, context_embeddings):
    # Encode the question
    question_inputs = question_encoder_tokenizer(question, return_tensors='pt', truncation=True, padding=True)
    question_embedding = question_encoder(**question_inputs).pooler_output
    
    # Compute dot product similarity
    similarities = torch.matmul(question_embedding, context_embeddings.T)
    best_match_idx = torch.argmax(similarities, dim=1).item()
    
    # Retrieve the best context
    best_context = contexts[best_match_idx]
    
    # Generate the answer using BART
    generator_inputs = generator_tokenizer(question + " " + best_context, return_tensors='pt', truncation=True, padding=True)
    summary_ids = generator_model.generate(generator_inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
    answer = generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return answer

# Streamlit App
st.title("RAG-Powered Insurance Chatbot")

st.write("""
    This chatbot provides answers based on an insurance policy document. 
    Enter your question below and get an answer generated by our AI model.
""")

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput input {
        background-color: #ffffff;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner('Generating answer...'):
            answer = generate_answer(question, context_embeddings)
        st.write("**Answer:**", answer)
    else:
        st.write("Please enter a question.")

