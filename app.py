import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Single AI Agent",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 Single AI Agent")

st.write(
    "AI Agent powered by Groq, Tavily Search and Weatherstack."
)


# --------------------------------------------------
# CHECK API KEYS
# --------------------------------------------------

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing from your .env file.")

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY is missing from your .env file.")

if not WEATHERSTACK_API_KEY:
    st.warning("⚠️ WEATHERSTACK_API_KEY is missing. Weather queries will not work.")


# --------------------------------------------------
# TAVILY SEARCH TOOL
# --------------------------------------------------

Search_tool = TavilySearch(
    max_results=3
)


# --------------------------------------------------
# WEATHER TOOL
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not set."

    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}: {data}"

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )

    except Exception as e:
        return f"Weather API error: {str(e)}"


# --------------------------------------------------
# CREATE LLM
# --------------------------------------------------

@st.cache_resource
def create_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY
    )


# --------------------------------------------------
# CREATE AGENT
# --------------------------------------------------

@st.cache_resource
def create_ai_agent():

    llm = create_llm()

    tools = [
        Search_tool,
        get_weather
    ]

    agent = create_agent(
        model=llm,
        tools=tools
    )

    return agent


# --------------------------------------------------
# INITIALIZE AGENT
# --------------------------------------------------

if GROQ_API_KEY and TAVILY_API_KEY:

    agent = create_ai_agent()

    # --------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # --------------------------------------------------
    # DISPLAY PREVIOUS MESSAGES
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # --------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------

    user_input = st.chat_input(
        "Ask me anything..."
    )


    if user_input:

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        # --------------------------------------------------
        # AGENT RESPONSE
        # --------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("🤔 Thinking..."):

                try:

                    response = agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_input
                                }
                            ]
                        }
                    )

                    answer = response["messages"][-1].content

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error: {str(e)}"
                    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🛠️ Available Tools")

    st.write("### 🔎 Tavily Search")
    st.write(
        "Searches the web for current information."
    )

    st.write("### 🌤️ Weather")
    st.write(
        "Gets current weather information for a city."
    )

    st.divider()

    st.write("### 🤖 AI Model")

    st.write(
        "Groq - openai/gpt-oss-20b"
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()