import logging
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import llm, tts, stt, inference
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
import time
from livekit.agents import function_tool, RunContext
from livekit.agents import AgentTask
from pypdf import PdfReader
import gspread
from oauth2client.service_account import ServiceAccountCredentials

logger = logging.getLogger(__name__)
load_dotenv()

def load_hospital_catalog(pdf_path: str = "hospital_catalog.pdf") -> str:
    """Extract text from a PDF file to use as the hospital service catalog."""
    if not os.path.exists(pdf_path):
        logger.warning(f"Catalog file not found at {pdf_path}. Using default catalog.")
        return "Cardiology, Neurology, Pediatrics, Emergency, Radiology, Orthopedics"
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return "Cardiology, Neurology, Pediatrics, Emergency, Radiology, Orthopedics"

DB_PATH = "patient_intake.db"

def init_db():
    """Create database and table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_intakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            full_name TEXT,
            age INTEGER,
            gender TEXT,
            city TEXT,
            phone_number TEXT,
            medical_issue TEXT,
            service_available BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)

def store_patient_intake(session_id: str, patient_data: dict):
    """Insert patient information into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO patient_intakes (
            session_id, full_name, age, gender, city, phone_number,
            medical_issue, service_available
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id,
        patient_data.get('full_name'),
        patient_data.get('age'),
        patient_data.get('gender'),
        patient_data.get('city'),
        patient_data.get('phone_number'),
        patient_data.get('medical_issue'),
        patient_data.get('service_available', False)
    ))
    conn.commit()
    conn.close()

class CollectConsent(AgentTask[bool]):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions="Ask for recording consent and get a clear yes or no answer. Be polite.",
            chat_ctx=chat_ctx,
        )
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Briefly introduce yourself, then ask for permission to record the call for quality assurance. Make it clear they can decline."
        )
    @function_tool
    async def consent_given(self) -> None:
        self.complete(True)
    @function_tool
    async def consent_denied(self) -> None:
        self.complete(False)

class Manager(Agent):
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions="You are a manager for a team of voice AI assistants. Handle escalations professionally.",
            tts="inworld/inworld-tts-1:ashley",
            chat_ctx=chat_ctx,
        )
    @function_tool
    async def escalate_to_manager(self, context: RunContext) -> "Manager":
        return Manager(chat_ctx=self.chat_ctx), "Escalating you to my manager now."

class Assistant(Agent):
    def __init__(self) -> None:
        catalog_text = load_hospital_catalog("hospital_catalog.pdf")
        instructions = f"""
        You are Sarah, a friendly and professional hospital intake assistant.
        
        **Workflow:**
        1. First, greet the caller and ask for their **full name**.
        2. Then ask for **age**, **gender**, **city**, and **phone number** (one at a time).
        3. After collecting all details, ask: "What medical issue brings you to the hospital today?"
        4. Based on the caller's answer, check if the issue is in the hospital's catalog below.
        5. If the issue matches any service in the catalog, inform them that a specialist will contact them soon.
        6. If not, explain that you will create a new case and someone will call back within 24 hours.
        
        **Hospital Catalog (services we offer):**
        {catalog_text}
        
        **Important rules:**
        - Do not invent services not listed.
        - Be empathetic and clear.
        - Collect all patient information before calling the `record_patient_info` tool.
        - Use the `check_service_availability` tool to verify if the issue is covered.
        """
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        consent = await CollectConsent(chat_ctx=self.chat_ctx)
        if consent:
            await self.session.generate_reply(
                instructions="Start the hospital intake process by asking for the patient's full name."
            )
        else:
            await self.session.generate_reply(
                instructions="Inform the user that you cannot proceed without consent and politely end the call."
            )

    @function_tool
    async def check_service_availability(self, context: RunContext, medical_issue: str) -> bool:
        """
        Check if the medical issue is covered by the hospital catalog.
        Returns True if available, False otherwise.
        """
        catalog = load_hospital_catalog("hospital_catalog.pdf")
        issue_lower = medical_issue.lower()
        keywords = ["cardio", "heart", "neuro", "brain", "pediatric", "child", "emergency", "radiology", "xray", "ortho", "bone"]
        available = any(kw in issue_lower for kw in keywords)
        for service in ["cardiology", "neurology", "pediatrics", "emergency", "radiology", "orthopedics"]:
            if service in issue_lower:
                available = True
                break
        return available

    @function_tool
    async def record_patient_info(
        self,
        context: RunContext,
        full_name: str,
        age: int,
        gender: str,
        city: str,
        phone_number: str,
        medical_issue: str
    ) -> None:
        """
        Call this tool after collecting all patient information.
        It stores the data in the database and provides a final response.
        """
        service_available = await self.check_service_availability(context, medical_issue)
        
        patient_data = {
            'full_name': full_name,
            'age': age,
            'gender': gender,
            'city': city,
            'phone_number': phone_number,
            'medical_issue': medical_issue,
            'service_available': service_available
        }
        session_id = context.session.id
        store_patient_intake(session_id, patient_data)
        append_to_google_sheet(patient_data) 
        if service_available:
            await context.session.say(
                "Thank you. I've noted all your details. A specialist from the relevant department will be in touch with you shortly. Is there anything else I can help you with today?"
            )
        else:
            await context.session.say(
                "Thank you for your patience. I have created a new case for you. Our case management team will review it and call you back at the number you provided within 24 hours. Is there anything else I can help you with?"
            )
def append_to_google_sheet(patient_data: dict):
    """Append one row to your Google Sheet."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open("Hospital Intakes").sheet1

    row = [
        patient_data.get('full_name', ''),
        patient_data.get('age', ''),
        patient_data.get('gender', ''),
        patient_data.get('city', ''),
        patient_data.get('phone_number', ''),
        patient_data.get('medical_issue', ''),
        "Yes" if patient_data.get('service_available') else "No",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
    sheet.append_row(row)
    print(f"✅ Data appended to Google Sheet: {row}")
server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    init_db()
    
    session = AgentSession(
        stt=stt.FallbackAdapter([
            inference.STT.from_model_string("deepgram/nova-3"),
            inference.STT.from_model_string("assemblyai/universal-streaming:en"),
        ]),
        llm=llm.FallbackAdapter([
            inference.LLM(model="openai/gpt-4.1-mini"),
            inference.LLM(model="google/gemini-2.5-flash-001"),
        ]),
        tts=tts.FallbackAdapter([
            inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
            inference.TTS.from_model_string("inworld/inworld-tts-1"),
        ]),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()
    last_eou_metrics = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("%s", summary)

    ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking" and last_eou_metrics:
            elapsed = time.time() - last_eou_metrics.timestamp
            logger.info("%.3f", elapsed)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)