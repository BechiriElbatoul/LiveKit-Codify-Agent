import logging
from dotenv import load_dotenv
from livekit import agents
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
import httpx
from livekit.agents import function_tool, RunContext, ToolError
from livekit.agents import AgentTask

logger = logging.getLogger(__name__)

load_dotenv()

class CollectConsent(AgentTask[bool]):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions="""
            Ask for recording consent and get a clear yes or no answer.
            Be polite and professional.
            """,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="""
            Briefly introduce yourself, then ask for permission to record the call for quality assurance and training purposes.
            Make it clear that they can decline.
            """
        )

    @function_tool
    async def consent_given(self) -> None:
        """Use this when the user gives consent to record."""
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Use this when the user denies consent to record."""
        self.complete(False)

class Manager(Agent):
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                "You are a manager for a team of helpful voice AI assistants. "
                "Handle escalations professionally."
            ),
            tts="inworld/inworld-tts-1:ashley",
            chat_ctx=chat_ctx,
        )
    @function_tool
    async def escalate_to_manager(self, context: RunContext) -> "Manager":
        """Escalate the call to a manager on user request."""
        return Manager(chat_ctx=self.chat_ctx), "Escalating you to my manager now."


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                """You are a frendly customer service representative.Help customers with general
                inquiries. If they ask for a manager or you cant resolve their issue,
                use the escalate_to_manager tool"""
            ),
        )

    async def on_enter(self) -> None:
        consent = await CollectConsent(chat_ctx=self.chat_ctx)

        if consent:
            await self.session.generate_reply(instructions="Offer your assistance to the user.")
        else:
            await self.session.generate_reply(instructions="Inform the user that you are unable to proceed and will end the call.")
    
    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict:
        """Look up current weather for a location
        
        Args: 
            location: City name or location to get weather for.
        """
        await context.session.say("Let me search for that...")        
        context.disallow_interruptions()

        async with httpx.AsyncClient() as client:
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                raise ToolError(f"Could not find location: {location}")

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            place_name = geo_data["results"][0]["name"]

            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code",  
                    "temperature_unit": "fahrenheit",
                }
            )
            weather = weather_response.json()

            return {
                "location": place_name,
                "temperature_f": weather["current"]["temperature_2m"],
                "conditions": weather["current"]["weather_code"],
                
            }

server = AgentServer() 

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=stt.FallbackAdapter(
            [
            inference.STT.from_model_string("deepgram/nova-3"),
            inference.STT.from_model_string("assemblyai/universal-streaming:en"),
            ]
        ),
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model="openai/gpt-4.1-mini"),
                inference.LLM(model="google/gemini-2.5-flash-001"),
            ]        
        ),
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation = True,
    )       

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOMetrics | None = None

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
        if ev.new_state == "speaking":
            if last_eou_metrics:
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