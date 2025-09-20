#!/usr/bin/env python3
import asyncio
from dotenv import load_dotenv
from agent import OptimizedVoiceAgent, create_agent_config

async def main():
    load_dotenv()
    
    # Test different configurations
    configs_to_test = ["default", "vad_fast", "vad_conservative", "model_based"]
    
    for config_name in configs_to_test:
        print(f"Testing configuration: {config_name}")
        config = create_agent_config(config_name)
        agent = OptimizedVoiceAgent(config)
        
        # Initialize session
        session = agent.initialize_session()
        print(f"✅ {config_name} configuration initialized successfully")
        print(f"   Turn detection: {config.turn_detection}")
        print(f"   Min delay: {config.min_endpointing_delay}s")
        print(f"   Max delay: {config.max_endpointing_delay}s")
        print()

if __name__ == "__main__":
    asyncio.run(main())