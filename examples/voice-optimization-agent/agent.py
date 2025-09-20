from livekit.agents import Agent, AgentSession, JobContext, RunContext, function_tool, cli, WorkerOptions
from livekit.agents import Agent, AgentSession, JobContext, RunContext, function_tool, cli, WorkerOptions
from livekit.plugins.deepgram import STT as DeepgramSTT
from livekit.plugins.cartesia import TTS as CartesiaTTS
from livekit.plugins.silero import VAD as SileroVAD
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.openai import LLM as OpenaiLLM  # Add this import
import os
from dotenv import load_dotenv

load_dotenv()

# Your existing tools and class here...

class FallbackTTS:
    """TTS wrapper that falls back from Cartesia to OpenAI TTS if Cartesia fails"""
    
    def __init__(self):
        self.cartesia_tts = None
        self.openai_tts = None
        self.use_openai = False
        
        # Try to initialize Cartesia TTS
        try:
            self.cartesia_tts = CartesiaTTS(
                api_key=os.getenv("CARTESIA_API_KEY"),
                model="sonic-2",
                voice="c5a1e070-7b8a-4b2b-9c5c-0b6a7b7b7b7b"
            )
            print("Initialized Cartesia TTS")
        except Exception as e:
            print(f"Failed to initialize Cartesia TTS: {e}")
            self.use_openai = True
            
        # Initialize OpenAI TTS as backup
        try:
            from livekit.plugins.openai import TTS as OpenAITTS
            self.openai_tts = OpenAITTS(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="tts-1",
                voice="alloy"
            )
            print("Initialized OpenAI TTS as backup")
        except Exception as e:
            print(f"Failed to initialize OpenAI TTS: {e}")
            
        if self.use_openai:
            print("Using OpenAI TTS")
        else:
            print("Using Cartesia TTS (with OpenAI fallback)")
    
    async def synthesize(self, text: str):
        """Synthesize text to audio, falling back to OpenAI if Cartesia fails"""
        if self.use_openai or self.cartesia_tts is None:
            # Use OpenAI TTS
            if self.openai_tts is None:
                raise Exception("No TTS available")
            return await self.openai_tts.synthesize(text)
        
        try:
            # Try Cartesia TTS first
            return await self.cartesia_tts.synthesize(text)
        except Exception as e:
            print(f"Cartesia TTS failed: {e}, switching to OpenAI TTS")
            self.use_openai = True
            
            # Try OpenAI TTS
            if self.openai_tts is None:
                raise Exception("Cartesia TTS failed and no OpenAI TTS available")
                
            return await self.openai_tts.synthesize(text)
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Initialize components with error handling
    try:
        print("Initializing STT...")
        stt = DeepgramSTT(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="nova",
        )
        
        print("Initializing TTS...")
        # Use FallbackTTS for robust TTS handling
        tts = FallbackTTS()
        
        print("Initializing VAD...")
        vad = SileroVAD.load()
        
        print("Initializing turn detector...")
        turn_detector = MultilingualModel()
        
        print("Initializing LLM...")
        llm = OpenaiLLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )
        
        print("Creating agent session...")
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            turn_detection=turn_detector,
            min_endpointing_delay=0.5,
            max_endpointing_delay=6.0,
            allow_interruptions=True
        )
        
        print("Creating agent...")
        agent = Agent(
            instructions="You are an optimized voice assistant with tuned parameters for better performance.",
            tools=[]
        )
        
        print("Starting agent session...")
        await session.start(agent=agent, room=ctx.room)
        
        print("Agent started successfully! Waiting for interactions...")
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))