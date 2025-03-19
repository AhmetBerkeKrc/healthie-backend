from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
import smtplib
from langchain_core.messages import  ToolMessage
from fastapi.responses import JSONResponse
app = FastAPI()

# Define the SMS tool using the @tool decorator
def load_appointment_list():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file("appointment_credential.json", scopes=scopes)
    client = gspread.authorize(creds)
    sheet_id = "11dSw6vLfkGpqeg3nApvGo4bM7Mwp9eQ_EOKog7Q76-M"
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet("app_sheet")
    values = worksheet.get_all_values()
    columns = values[0]   
    return pd.DataFrame(values[1:], columns=columns), worksheet

# Email sending function
def send_appointment_email(patient_id, patient_name, patient_email, doctor_name, time):
    
    df, worksheet = load_appointment_list()

    patient_info = df[df["Patient ID Number"] == patient_id]

    if patient_info.empty:
        return "No appointment found for the given ID."

    subject = "Appointment Information"
    body = f"""
        Dear {patient_name},

        Your appointment details are as follows:

        - Patient ID Number: {patient_id}
        - Phone Number: {patient_info["Patient Phone Number"].values[0]}
        - Doctor: Dr. {doctor_name}
        - Appointment Date & Time: {patient_info["Date"].values[0]} - {time}

        If you have any questions, please contact our office.

        Best regards,  
        Your Healthcare Team
    """

    sender_email = "ahmetberke.karaca@gmail.com"
    sender_password = "nbqo xfiy sihi lcut"  
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  

    try:
        # Establish connection to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure connection
        server.login(sender_email, sender_password)

        # Create email message
        message = f"Subject: {subject}\n\n{body}"

        # Send email
        server.sendmail(sender_email, patient_email, message)
        server.quit()
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"

# Existing code



@tool
def available_hours(listing_request: str) -> str:
    """
    Fetches and returns available appointment hours from a Google Sheets schedule.

    Args:
        listing_request (str): User's request to see available hours.

    Returns:
        str: A list of available appointment slots.
    """
    try:
        df, worksheet = load_appointment_list()

        available_hours = df[df["Status"] == "Available"].reset_index(drop=True)

        return available_hours[["Doctor Name", "Date", "Time"]].to_dict(orient="records")
    
    except Exception as e:
        return f"Failed to retrieve the data: {e}"

# Book an appointment for Ahmet Berke Karaca with ID 21496257938, phone number +905348620505, and email ahmetberke.karaca@outlook.com from Dr. Mehmet Oz at 13:00
@tool 
def book_appointment(doctor_name: str, time: str, patient_name: str, patient_id: str, patient_number: str, patient_email: str) -> str:
    """
    Book an appointment for {patient_name} with ID {patient_id}, phone number {patient_number}, and email {patient_email} from Dr. {doctor_name} at {time}.
    
    Args:
        doctor_name (str): Doctor's name.
        time (str): Appointment time (HH:MM format).
        patient_name (str): Patient's full name.
        patient_id (str): Patient's ID number.
        patient_number (str): Patient's phone number.
        patient_email (str): Patient's email address.

    Returns:
        str: Result of the booking process.
    """
    try:
        # Ensure all required fields are provided
        if not all([doctor_name, time, patient_name, patient_id, patient_number, patient_email]):
            return "Booking failed: All fields (doctor_name, time, patient_name, patient_id, patient_number, patient_email) must be provided."
        
        df, worksheet = load_appointment_list()

        # Check if the requested appointment slot is available
        mask = (df["Doctor Name"] == doctor_name) & (df["Time"] == time) & (df["Status"] == "Available")

        if not mask.any():
            return f"Appointment booking failed: No available slots for Dr. {doctor_name} at {time}."

        # Update the appointment slot with the patient's details
        df.loc[mask, "Patient Full Name"] = patient_name
        df.loc[mask, "Patient ID Number"] = patient_id
        df.loc[mask, "Patient Phone Number"] = patient_number
        df.loc[mask, "Patient Email"] = patient_email
        df.loc[mask, "Status"] = "Booked"

        # Update the Google Sheets with the new appointment data
        updated_values = [df.columns.tolist()] + df.values.tolist()
        worksheet.update(values=updated_values, range_name="A1")

        # Send the confirmation email after booking
        email_result = send_appointment_email(patient_id, patient_name, patient_email, doctor_name, time)
        
        return f"Appointment successfully booked for {patient_name} with Dr. {doctor_name} at {time}. {email_result}"

    except Exception as e:
        return f"Booking failed: {e}"

tool_mapping = {"available_hours": available_hours, "book_appointment": book_appointment}


###----------------------------- AI model -----------------------------###
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6,
    openai_api_key="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"
)

llm_with_tools = llm.bind_tools([available_hours, book_appointment])

template = PromptTemplate(
    input_variables=['context', 'user_input'],
    template="""
    You are a professional medical chatbot designed to assist users with health-related queries and providing the current available hours in the schedule. Your name is HE4LTHY. 
    
    ### Guidelines:
    - Use emojis to make your service more engaging, but don't use too much.
    - Provide **clear, concise, and medically accurate** responses.
    - Use the provided **context** as the base for your answer.
    - **Do not diagnose, prescribe medications, or suggest treatments** beyond general health advice.
    - **Health is crucial**, so avoid absolute statements; instead, **encourage consulting a healthcare professional** when needed.
    - If the user greets you, respond appropriately with a greeting.
    - If the user asks about **non-medical topics**, respond strictly with: "I can only assist you with your health-related questions appointment booking operations."
    - If the user asks for the **current available hours in the schedule**, retrieve and provide the available slots from the Google Sheets schedule.
    - If the user wants to **book an appointment**, update columns accordingly with the given arguments from Google Sheets schedule and update it on Google Sheets.

    ### Context:
    {context}

    ### User Input:
    {user_input}
    """
)
model_chain = template | llm_with_tools


###----------------------------- VECTOR DB -----------------------------###
st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

faiss_index = faiss.read_index("medical_faiss.index")

dataset = load_dataset("codexist/medical_data")
texts = [row["data"] for row in dataset["train"]]

###----------------------------- LOOP -----------------------------###
def func(user_input: str):
    query_vector = st_model.encode([user_input]).astype(np.float32)
    distances, indices = faiss_index.search(query_vector, 5)
    relevant_texts = [texts[idx] for idx in indices[0]]
    context = "\n".join(relevant_texts)
    response = model_chain.invoke({'context': context, 'user_input': user_input})
    y = None
    for tool_call in response.tool_calls:
        tool = tool_mapping[tool_call["name"].lower()]
        tool_output = tool.invoke(tool_call["args"])
        X = ToolMessage(tool_output, tool_call_id=tool_call["id"])
        y = X.content
    if y is None:
        return response.content
    else:
        return y

# uvicorn langchain_:app --reload

class UserInput(BaseModel):
    user_prompt: str


@app.post("/chat")
async def chat(user_input: UserInput):
    result = func(user_input.user_prompt)
    print(result)

    # Eğer başka bir formatta ise, text/plain olarak döndürebiliriz
    return JSONResponse(content={"result": result})

@app.get("/available-hour")
async def available_hour():
    result = func('available hours')
    # Assuming the result is a list, we can directly return it as JSON
    return JSONResponse(content={"available_hours": result})

@app.post("/book")
async def book(user_input: UserInput):
    result = func(user_input.user_prompt)
    
    print(result)
    return JSONResponse(content={"result": result})
