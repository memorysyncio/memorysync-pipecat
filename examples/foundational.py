#
# Foundational example: a voice agent that remembers callers across calls.
#
# MemorySyncMemoryService sits between the user context aggregator and the
# LLM: every LLMContextFrame is enriched with relevant long-term memories
# under a hard time budget, and the caller's new turns are sent to fact
# extraction in the background (only the durable facts are stored; the
# bot's replies are not). A slow or unreachable memory backend can never
# stall the voice reply.
#
# Run (choose any transport supported by the Pipecat runner):
#
#   uv add pipecat-memorysync "pipecat-ai[deepgram,cartesia,openai,silero,runner,webrtc]"
#   export MEMORYSYNC_API_KEY=ms_...   # https://app.memorysync.io
#   export DEEPGRAM_API_KEY=...
#   export CARTESIA_API_KEY=...
#   export OPENAI_API_KEY=...
#   python foundational.py
#
# Then open http://localhost:7860/client, talk to the bot, tell it your
# name and a preference, hang up, and call again: it remembers.
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat_memorysync import MemorySyncMemoryService

load_dotenv(override=True)

SYSTEM_INSTRUCTION = (
    "You are a friendly voice assistant with long-term memory. "
    "Relevant facts about the caller may appear as background memory context. "
    "Use them naturally; never read them out verbatim. "
    "Your answers are spoken aloud, so keep them short and conversational."
)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
    )

    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])

    # Long-term memory. In a real deployment, derive user_id from the
    # caller's identity (phone number, account id, ...) so each caller
    # gets their own memories.
    memory = MemorySyncMemoryService(
        api_key=os.environ["MEMORYSYNC_API_KEY"],
        user_id=os.environ.get("MEMORYSYNC_USER_ID", "demo-caller"),
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_INSTRUCTION}])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            memory,  # enrich with memories + send the caller's new turns, on budget
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with the Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
