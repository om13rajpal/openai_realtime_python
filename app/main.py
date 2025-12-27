import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import get_settings
from app.redis_client import init_redis, close_redis
from app.models import (
    SessionCreateRequest,
    SessionCreateResponse,
    WebSocketMessage,
    MaritimeChatRequest,
    MaritimeChatResponse,
    StructuredChatResponse,
    ToolCallInfo,
    AddVesselDetailsRequest,
    AddVesselDetailsResponse,
    VesselDetails,
)
from app.session_manager import (
    create_session,
    get_session,
    get_conversation,
    add_to_conversation,
    delete_session,
    delete_conversation,
    update_session_status,
)
from app.realtime_bridge import RealtimeBridge
from app.tools import MARITIME_TOOLS, execute_tool
from app.system_prompt import MARITIME_SAFETY_PROMPT
from app.external_api import format_user_context, fetch_user_settings, fetch_weather_data
from app.response_formatter import (
    format_response_from_tool_calls,
    format_normal_response,
    create_streaming_chunk,
)

logger = logging.getLogger(__name__)

# External API base URL for vessel management
EXTERNAL_API_BASE = "http://52.5.27.89:3000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    await init_redis()
    yield
    # Shutdown
    await close_redis()


app = FastAPI(
    title="OpenAI Realtime Bridge",
    description="WebSocket bridge for OpenAI Realtime API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/session", response_model=SessionCreateResponse)
async def create_new_session(request: SessionCreateRequest = SessionCreateRequest()):
    """Create a new session for WebSocket connection."""
    settings = get_settings()
    base_url = f"wss://{settings.host}:{settings.port}" if settings.host != "0.0.0.0" else ""

    result = await create_session(request, base_url)
    return SessionCreateResponse(**result)


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for realtime communication."""
    # Validate token
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    session = await get_session(token)
    if not session:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # Accept connection
    await websocket.accept()

    # Update session status
    await update_session_status(token, "connected")

    # Create bridge
    bridge = RealtimeBridge(
        voice=session.voice,
        instructions=session.instructions,
        conversation_id=session.conversation_id
    )

    async def send_to_client(message: dict):
        """Send message to WebSocket client."""
        try:
            await websocket.send_json(message)
        except Exception:
            pass

    # Start tasks
    bridge_task = None
    try:
        # Start OpenAI connection in background
        bridge_task = asyncio.create_task(bridge.start(send_to_client))

        # Handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "audio":
                    await bridge.send_audio(message.get("data", ""))

                elif msg_type == "commit":
                    await bridge.commit_audio()

                elif msg_type == "cancel":
                    await bridge.cancel_response()

                elif msg_type == "close":
                    # Immediate cleanup
                    await delete_conversation(session.conversation_id)
                    break

            except json.JSONDecodeError:
                # Ignore malformed messages
                pass

    except WebSocketDisconnect:
        pass

    finally:
        # Cleanup
        await bridge.disconnect()
        if bridge_task:
            bridge_task.cancel()
            try:
                await bridge_task
            except asyncio.CancelledError:
                pass

        await delete_session(token, session.conversation_id)


@app.post("/maritime-chat", response_model=StructuredChatResponse)
async def maritime_chat(request: MaritimeChatRequest):
    """
    Maritime chat endpoint with tool-augmented LLM responses.

    Returns structured JSON responses with type field:
    - weather: includes 'report' field with weather details
    - assistance: includes 'local_assistance' array
    - route: includes 'trip_plan' with route and analysis
    - normal: just message text

    This endpoint:
    1. Receives user messages about maritime queries
    2. Uses LLM with function calling to determine which tools to use
    3. Executes tools (weather, local assistance, route planning)
    4. Returns structured AI response with typed data
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        # Build context from user data
        user_settings = None
        weather_data = None

        if request.auth_token:
            user_settings = await fetch_user_settings(request.auth_token)
            if request.coordinates:
                weather_data = await fetch_weather_data(request.auth_token, request.coordinates)

        # Format user context
        context = format_user_context(user_settings, weather_data, request.user_location)

        # Add vessel info to context if provided
        if request.vessels:
            vessel_info = [
                {"make": v.get("make", "Unknown"), "model": v.get("model", "Unknown"), "year": v.get("year", "Unknown")}
                for v in request.vessels
            ]
            context += f"\n[Vessel Info: {vessel_info}]"

        # Build system prompt with context
        system_prompt = f"{context}\n\n{MARITIME_SAFETY_PROMPT}" if context else MARITIME_SAFETY_PROMPT

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if request.conversation_history:
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        # Call OpenAI with tools
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=MARITIME_TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message
        tool_calls_made = []
        route_data = None

        # Process tool calls if any
        if assistant_message.tool_calls:
            # Execute each tool call
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                result = await execute_tool(tool_name, tool_args, request.auth_token)

                tool_calls_made.append({
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "result": result
                })

                # Check for route data
                if tool_name == "plan_and_analyze_marine_route" and result.get("success"):
                    route_data = result

                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result)
                })

            # Add assistant message with tool calls to messages
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Add tool results
            messages.extend(tool_results)

            # Get final response with tool results
            final_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            final_message = final_response.choices[0].message.content or ""
        else:
            final_message = assistant_message.content or ""

        # Format response using the response formatter
        structured_response = format_response_from_tool_calls(
            message=final_message or "I'm here to help with maritime questions.",
            tool_calls=tool_calls_made,
            route_data=route_data
        )

        return StructuredChatResponse(**structured_response)

    except Exception as e:
        logger.error("Maritime chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maritime-chat/stream")
async def maritime_chat_stream(request: MaritimeChatRequest):
    """
    Streaming maritime chat endpoint with real-time JSON updates.

    Returns Server-Sent Events (SSE) with partial JSON updates as they become available.
    Each event contains a delta with the updated field and value.

    Event format:
    - type: "delta" | "complete" | "error"
    - field: field name being updated
    - value: new value for the field
    - response_type: weather | assistance | route | normal
    """

    async def generate_stream() -> AsyncGenerator[str, None]:
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        try:
            # Build context from user data
            user_settings = None
            weather_data = None

            if request.auth_token:
                user_settings = await fetch_user_settings(request.auth_token)
                if request.coordinates:
                    weather_data = await fetch_weather_data(request.auth_token, request.coordinates)

            # Format user context
            context = format_user_context(user_settings, weather_data, request.user_location)

            # Add vessel info to context if provided
            if request.vessels:
                vessel_info = [
                    {"make": v.get("make", "Unknown"), "model": v.get("model", "Unknown"), "year": v.get("year", "Unknown")}
                    for v in request.vessels
                ]
                context += f"\n[Vessel Info: {vessel_info}]"

            # Build system prompt with context
            system_prompt = f"{context}\n\n{MARITIME_SAFETY_PROMPT}" if context else MARITIME_SAFETY_PROMPT

            # Build messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history if provided
            if request.conversation_history:
                for msg in request.conversation_history:
                    messages.append({"role": msg.role, "content": msg.content})

            # Add current user message
            messages.append({"role": "user", "content": request.message})

            # First call to determine tool usage
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=MARITIME_TOOLS,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message
            tool_calls_made = []
            route_data = None

            # Process tool calls if any
            if assistant_message.tool_calls:
                # Send processing status
                yield f"data: {json.dumps({'type': 'status', 'status': 'processing_tools'})}\n\n"

                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Send tool execution status
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

                    # Execute the tool
                    result = await execute_tool(tool_name, tool_args, request.auth_token)

                    tool_calls_made.append({
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": result
                    })

                    # Check for route data
                    if tool_name == "plan_and_analyze_marine_route" and result.get("success"):
                        route_data = result

                    # Send tool result preview
                    yield f"data: {json.dumps({'type': 'tool_complete', 'tool': tool_name, 'success': result.get('success', False)})}\n\n"

                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": json.dumps(result)
                    })

                # Add messages for final response
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                messages.extend(tool_results)

                # Stream final response
                yield f"data: {json.dumps({'type': 'status', 'status': 'generating_response'})}\n\n"

                stream = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=True
                )

                full_message = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_message += content
                        yield f"data: {json.dumps({'type': 'message_delta', 'content': content})}\n\n"

                # Format and send final structured response
                structured_response = format_response_from_tool_calls(
                    message=full_message or "I'm here to help with maritime questions.",
                    tool_calls=tool_calls_made,
                    route_data=route_data
                )

                yield f"data: {json.dumps({'type': 'complete', 'response': structured_response})}\n\n"

            else:
                # Stream response directly
                yield f"data: {json.dumps({'type': 'status', 'status': 'generating_response'})}\n\n"

                stream = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=True
                )

                full_message = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_message += content
                        yield f"data: {json.dumps({'type': 'message_delta', 'content': content})}\n\n"

                # Format and send final response
                structured_response = format_normal_response(
                    message=full_message or "I'm here to help with maritime questions."
                )

                yield f"data: {json.dumps({'type': 'complete', 'response': structured_response})}\n\n"

        except Exception as e:
            logger.error("Streaming chat error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/chat/with-context")
async def chat_with_context(request: MaritimeChatRequest, conversation_id: Optional[str] = Query(None)):
    """
    Chat endpoint that maintains context across text and voice sessions.

    This endpoint allows sharing conversation history between:
    - Text-based chat (this endpoint)
    - Voice-based chat (WebSocket)

    If a conversation_id is provided, it will:
    1. Load existing conversation history from Redis
    2. Add the new message and response to history
    3. Return the structured response

    This enables seamless switching between voice and text modes
    while maintaining full conversation context.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        # Build context from user data
        user_settings = None
        weather_data = None

        if request.auth_token:
            user_settings = await fetch_user_settings(request.auth_token)
            if request.coordinates:
                weather_data = await fetch_weather_data(request.auth_token, request.coordinates)

        # Format user context
        context = format_user_context(user_settings, weather_data, request.user_location)

        # Add vessel info to context if provided
        if request.vessels:
            vessel_info = [
                {"make": v.get("make", "Unknown"), "model": v.get("model", "Unknown"), "year": v.get("year", "Unknown")}
                for v in request.vessels
            ]
            context += f"\n[Vessel Info: {vessel_info}]"

        # Build system prompt with context
        system_prompt = f"{context}\n\n{MARITIME_SAFETY_PROMPT}" if context else MARITIME_SAFETY_PROMPT

        # Build messages from Redis history if conversation_id provided
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_id:
            # Load conversation history from Redis
            conv_data = await get_conversation(conversation_id)
            if conv_data:
                for msg in conv_data.history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.transcript
                    })

        # Also add any conversation history from the request
        if request.conversation_history:
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        # Call OpenAI with tools
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=MARITIME_TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message
        tool_calls_made = []
        route_data = None

        # Process tool calls if any
        if assistant_message.tool_calls:
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = await execute_tool(tool_name, tool_args, request.auth_token)

                tool_calls_made.append({
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "result": result
                })

                if tool_name == "plan_and_analyze_marine_route" and result.get("success"):
                    route_data = result

                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result)
                })

            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            messages.extend(tool_results)

            final_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            final_message = final_response.choices[0].message.content or ""
        else:
            final_message = assistant_message.content or ""

        # Save to conversation history if conversation_id provided
        if conversation_id:
            await add_to_conversation(conversation_id, "user", request.message, "text")
            await add_to_conversation(conversation_id, "assistant", final_message, "text")

        # Format response
        structured_response = format_response_from_tool_calls(
            message=final_message or "I'm here to help with maritime questions.",
            tool_calls=tool_calls_made,
            route_data=route_data
        )

        # Add conversation_id to response for reference
        structured_response["conversation_id"] = conversation_id

        return StructuredChatResponse(**structured_response)

    except Exception as e:
        logger.error("Chat with context error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_vessel_details", response_model=AddVesselDetailsResponse)
async def add_vessel_details(request: AddVesselDetailsRequest):
    """
    Generate vessel details using LLM and add to user's account.

    This endpoint:
    1. Takes partial vessel info or text description
    2. Uses LLM to generate complete vessel details
    3. Calls external API to add vessel to user's account
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        # Build prompt for vessel details generation
        vessel_input = request.vessel_input
        input_text = ""

        if vessel_input.text_description:
            input_text = vessel_input.text_description
        else:
            parts = []
            if vessel_input.make:
                parts.append(f"Make: {vessel_input.make}")
            if vessel_input.model:
                parts.append(f"Model: {vessel_input.model}")
            if vessel_input.year:
                parts.append(f"Year: {vessel_input.year}")
            if vessel_input.vessel_type:
                parts.append(f"Type: {vessel_input.vessel_type}")
            input_text = ", ".join(parts) if parts else "Unknown vessel"

        prompt = f"""You are a marine vessel expert. Based on the following vessel information, provide complete and accurate vessel specifications.

Vessel Input: {input_text}

Provide a JSON response with the following structure. Use accurate data for the specific make/model if known, or reasonable estimates based on vessel type and year:

{{
    "make": "Manufacturer name",
    "model": "Model name",
    "year": "Year as string",
    "vessel_type": "sailboat|powerboat|yacht|catamaran|trawler|fishing|pontoon|jet_ski|other",
    "length_feet": 0.0,
    "beam_feet": 0.0,
    "draft_feet": 0.0,
    "displacement_lbs": 0,
    "engine_type": "outboard|inboard|sail|electric|hybrid",
    "fuel_capacity_gallons": 0.0,
    "water_capacity_gallons": 0.0,
    "max_speed_knots": 0.0,
    "cruising_speed_knots": 0.0,
    "sleeping_capacity": 0,
    "recommended_crew": 0,
    "safety_features": ["feature1", "feature2"],
    "suitable_conditions": "Description of suitable weather/sea conditions for this vessel"
}}

Be accurate for known vessels. For unknown vessels, provide reasonable estimates based on type and size."""

        # Generate vessel details using LLM
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a marine vessel expert. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        vessel_data = json.loads(response.choices[0].message.content)

        # Create VesselDetails object
        vessel_details = VesselDetails(
            make=vessel_data.get("make", vessel_input.make or "Unknown"),
            model=vessel_data.get("model", vessel_input.model or "Unknown"),
            year=vessel_data.get("year", vessel_input.year or "Unknown"),
            vessel_type=vessel_data.get("vessel_type"),
            length_feet=vessel_data.get("length_feet"),
            beam_feet=vessel_data.get("beam_feet"),
            draft_feet=vessel_data.get("draft_feet"),
            displacement_lbs=vessel_data.get("displacement_lbs"),
            engine_type=vessel_data.get("engine_type"),
            fuel_capacity_gallons=vessel_data.get("fuel_capacity_gallons"),
            water_capacity_gallons=vessel_data.get("water_capacity_gallons"),
            max_speed_knots=vessel_data.get("max_speed_knots"),
            cruising_speed_knots=vessel_data.get("cruising_speed_knots"),
            sleeping_capacity=vessel_data.get("sleeping_capacity"),
            recommended_crew=vessel_data.get("recommended_crew"),
            safety_features=vessel_data.get("safety_features"),
            suitable_conditions=vessel_data.get("suitable_conditions")
        )

        # Add vessel to user's account via external API
        vessel_id = None
        try:
            async with httpx.AsyncClient(timeout=15.0) as http_client:
                api_response = await http_client.post(
                    f"{EXTERNAL_API_BASE}/vessels",
                    headers={"Authorization": f"Bearer {request.auth_token}"},
                    json={
                        "make": vessel_details.make,
                        "model": vessel_details.model,
                        "year": vessel_details.year,
                        "vessel_type": vessel_details.vessel_type,
                        "length_feet": vessel_details.length_feet,
                        "beam_feet": vessel_details.beam_feet,
                        "draft_feet": vessel_details.draft_feet,
                        "displacement_lbs": vessel_details.displacement_lbs,
                        "engine_type": vessel_details.engine_type,
                        "fuel_capacity_gallons": vessel_details.fuel_capacity_gallons,
                        "water_capacity_gallons": vessel_details.water_capacity_gallons,
                        "max_speed_knots": vessel_details.max_speed_knots,
                        "cruising_speed_knots": vessel_details.cruising_speed_knots,
                        "sleeping_capacity": vessel_details.sleeping_capacity,
                        "recommended_crew": vessel_details.recommended_crew,
                        "safety_features": vessel_details.safety_features,
                        "suitable_conditions": vessel_details.suitable_conditions
                    }
                )

                if api_response.status_code in (200, 201):
                    result = api_response.json()
                    vessel_id = result.get("data", {}).get("id") or result.get("id")
                    return AddVesselDetailsResponse(
                        success=True,
                        message=f"Vessel {vessel_details.make} {vessel_details.model} ({vessel_details.year}) added successfully.",
                        vessel_details=vessel_details,
                        vessel_id=vessel_id
                    )
                else:
                    logger.warning("Vessel API returned status %s: %s", api_response.status_code, api_response.text)
                    return AddVesselDetailsResponse(
                        success=False,
                        message=f"Failed to add vessel to account: API returned {api_response.status_code}",
                        vessel_details=vessel_details
                    )

        except Exception as api_error:
            logger.error("Error calling vessel API: %s", api_error)
            return AddVesselDetailsResponse(
                success=False,
                message=f"Vessel details generated but failed to save: {str(api_error)}",
                vessel_details=vessel_details
            )

    except Exception as e:
        logger.error("Add vessel details error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
