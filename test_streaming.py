#!/usr/bin/env python3
"""
Test script to demonstrate LangGraph token streaming.

This simplified version shows how token streaming works in LangGraph.
"""

import asyncio
import os
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Simple example tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city} with a temperature of 72°F."

@tool
def calculate_area(length: float, width: float) -> str:
    """Calculate the area of a rectangle."""
    area = length * width
    return f"The area of a rectangle with length {length} and width {width} is {area} square units."

# Define the state
class AgentState(MessagesState):
    pass

# Agent node
async def agent(state: AgentState) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

# Bind tools to LLM
tools = [get_weather, calculate_area]
llm_with_tools = llm.bind_tools(tools)

# Build the graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)
workflow.set_entry_point("agent")

# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

async def demonstrate_streaming():
    """Demonstrate different streaming modes."""
    config = {"configurable": {"thread_id": "test_thread"}}

    print("=" * 60)
    print("LangGraph Token Streaming Demonstration")
    print("=" * 60)

    # Test queries
    queries = [
        "What's the weather in New York?",
        "Calculate the area of a rectangle with length 10 and width 5",
        "Tell me a short story about a robot learning to cook"
    ]

    for query in queries:
        print(f"\n\n📝 Query: {query}")
        print("-" * 40)

        # Track streaming state
        streaming_response = False

        # Stream with multiple modes
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(content=query)]},
            config,
            stream_mode=["messages", "updates", "custom"],
        ):
            # Handle token streaming (messages mode)
            if mode == "messages":
                try:
                    # Messages mode returns tuple: (message_chunk, metadata)
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        message_chunk, metadata = chunk

                        # Check if this is from the agent node (not tool output)
                        if metadata.get("langgraph_node") == "agent":
                            # Extract and print the content
                            if hasattr(message_chunk, 'content') and message_chunk.content:
                                if not streaming_response:
                                    print("\n🤖 Assistant: ", end="", flush=True)
                                    streaming_response = True
                                print(message_chunk.content, end="", flush=True)
                except Exception as e:
                    # Uncomment for debugging
                    # print(f"\n[Debug] Error: {e}")
                    pass

            # Handle node updates
            elif mode == "updates":
                # End streaming when we get updates
                if streaming_response:
                    print()  # New line after streamed response
                    streaming_response = False

                # Check for tool calls
                if isinstance(chunk, dict):
                    for node, update in chunk.items():
                        if node == "tools" and isinstance(update, dict):
                            msgs = update.get("messages", [])
                            for msg in msgs:
                                if hasattr(msg, 'content'):
                                    print(f"\n🔧 Tool Output: {msg.content}")

            # Handle custom data (if any)
            elif mode == "custom":
                if chunk:
                    print(f"\n📊 Custom Data: {chunk}")

        # Final newline if still streaming
        if streaming_response:
            print()

        # Small delay between queries
        await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("Streaming demonstration complete!")
    print("=" * 60)

async def main():
    """Main function to run the demo."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    try:
        await demonstrate_streaming()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())