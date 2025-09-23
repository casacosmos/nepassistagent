#!/usr/bin/env python3
"""
LangGraph Agent for NEPAssist Tanks Map Generation Chatbot.

This script sets up a LangGraph-based chatbot agent that:
- Engages in conversation with the user.
- Accepts a KMZ or GeoJSON file path via chat input.
- Uses the MCP Tool (from mcp_tool.py) to generate PDF maps when requested.
- Responds with generation status and output paths.

Requirements:
- langchain, langgraph, langchain_openai (or your preferred LLM provider).
- Set OPENAI_API_KEY environment variable for OpenAI LLM.

Run with: python this_script.py
Then interact via the console (input messages, receive responses).
"""

import os
from typing import Annotated, Dict, List
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Import the MCP tool's run function (assume mcp_tool.py is in the same directory)
from mcp_tool import run as mcp_run  # Replace with actual import if needed

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Use your preferred model

# Define the MCP Tool
@tool
def generate_map(
    input_file: str,
    images_dir: str = "images",
    output_dir: str = "output",
    out_stem: str = "NEPAssist_Tanks",
    source_type: str = "kmz",  # or "geojson"
    image_size: int = 0,  # 0 for auto
    map_style: str = "satellite",
    export_pdf: bool = True,
    export_geojson: bool = True,
    # Layout and rendering options (defaults apply unless overridden by chat)
    dpi: int = 300,
    layout_template: str = "Letter ANSI A Landscape",
    map_title: str = "",
    zoom_padding: float = 0.07,
    zoom_to_features: bool = True,
    hide_date: bool = True,
    show_north_arrow: bool = True,
    show_scale: bool = True,
    legend_position: str = "right",
    legend_hide_site: bool = False,
    legend_hide_buffer: bool = False,
    legend_hide_points: bool = False,
    legend_label_prefix: str = "Tank",
    legend_label_max_len: int = 40,
) -> str:
    """
    Generate a PDF map from a KMZ or GeoJSON file using the MCP Tool.

    Args:
        input_file: Path to the KMZ or GeoJSON file.
        images_dir: Directory containing marker images (default: "images").
        output_dir: Output directory for PDF and GeoJSON (default: "output").
        out_stem: Filename stem for outputs (default: "NEPAssist_Tanks").
        source_type: "kmz" or "geojson" (auto-detected if possible).
        image_size: Marker size in pixels (0 for auto-sizing).
        map_style: Base map style ("satellite" or "professional").
        export_pdf: Whether to export PDF (default: True).
        export_geojson: Whether to export GeoJSON (default: True).

    Returns:
        Status message with output paths.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        return f"Error: Input file '{input_file}' not found."

    # Prepare config dict matching MCP Tool's expectations
    cfg = {
        "source_type": source_type,
        "input_kmz": str(input_path) if source_type == "kmz" else None,
        "input_geojson": str(input_path) if source_type == "geojson" else None,
        "images_dir": images_dir,
        "output_filename_stem": out_stem,
        "image_size": image_size,
        "use_images": True,
        "map_style": map_style,
        "export_pdf": export_pdf,
        "export_geojson": export_geojson,
        # Top-level options recognized by mcp_tool
        "dpi": dpi,
        "zoom_padding": zoom_padding,
        # Nested map_options also recognized by mcp_tool
        "map_options": {
            "hide_date": hide_date,
            "show_north_arrow": show_north_arrow,
            "show_scale": show_scale,
            "legend_position": legend_position,
            "legend_hide_site": legend_hide_site,
            "legend_hide_buffer": legend_hide_buffer,
            "legend_hide_points": legend_hide_points,
            "legend_label_prefix": legend_label_prefix,
            "legend_label_max_len": legend_label_max_len,
            "zoom_to_features": zoom_to_features,
            "zoom_padding": zoom_padding,
            "layout_template": layout_template,
            "title_text": (map_title if str(map_title).strip() else " "),
        },
    }

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir_path / f"{out_stem}.pdf"
    geojson_path = out_dir_path / f"{out_stem}.geojson"

    try:
        mcp_run(cfg, out_dir_path)
    except Exception as e:
        return f"Error during map generation: {str(e)}"

    # Check if requested files were actually created
    status = "Map generation completed with the following results:\n"
    pdf_generated = export_pdf and pdf_path.exists()
    geojson_generated = export_geojson and geojson_path.exists()

    if pdf_generated:
        status += f"- PDF: {pdf_path.absolute()}\n"
    elif export_pdf:
        status += "- PDF: Failed to generate (check connectivity to ArcGIS service or logs for details)\n"

    if geojson_generated:
        status += f"- GeoJSON: {geojson_path.absolute()}\n"
    elif export_geojson:
        status += "- GeoJSON: Failed to generate\n"

    if (export_pdf and not pdf_generated) or (export_geojson and not geojson_generated):
        status = "Partial failure in generation.\n" + status
    else:
        status = "Map generation successful!\n" + status

    return status

# Bind the tool to the LLM
tools = [generate_map]
llm_with_tools = llm.bind_tools(tools)

# Define the state (using MessagesState for chat history)
class AgentState(MessagesState):
    pass  # Extends MessagesState for simplicity

# Agent node: Decide action based on LLM output
def agent(state: AgentState) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build the graph
workflow = StateGraph(state_schema=AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

# Add edges
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Routes to "tools" if tool calls, else END
    {
        "tools": "tools",
        END: END,
    },
)

# Set entry point
workflow.set_entry_point("agent")

# Compile with memory checkpoint
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Chatbot interaction loop
def chat():
    config = {"configurable": {"thread_id": "map_chat_thread"}}
    print("Chatbot ready! Type 'exit' to quit.")
    print("Example: 'Generate a map from /path/to/my.kmz' (defaults to satellite style).")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Invoke the graph
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        )
        
        # Print tool outputs and final agent response
        for event in events:
            if "messages" not in event or not event["messages"]:
                continue
            last_msg = event["messages"][-1]
            if last_msg.type == "tool":
                print(f"\nTool: {last_msg.content}")
            elif last_msg.type == "ai":
                print(f"\nAgent: {last_msg.content}")

if __name__ == "__main__":
    # Ensure required directories exist (optional)
    Path("images").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    chat()
