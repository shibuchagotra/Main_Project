import streamlit as st
import os
import json
import pandas as pd
import random
from datetime import datetime
from os.path import join

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from streamlit_feedback import streamlit_feedback

from PIL import Image
import time
import uuid
import asyncio

# Gemini API requires async
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
#functions for the graph
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")



async def ask_question(quest:str,model:str):
    """
    Creates a new thread, streams run events, and collects the latest state.

    Args:
        client: LangGraph SDK client
        assistant_id (str): Assistant/graph ID (e.g., "agent")
        input_data (dict): Input payload for the graph

    Returns:
        tuple: (thread_id, collected_state)
    """
    # Create a new thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    input_data = {
        "question": quest,
        "model":model
    }

    collected_state = {}

    # Stream events
    async for event in client.runs.stream(
        thread_id=thread_id,
        assistant_id='agent',
        input=input_data,
        stream_mode="values",
    ):
        if event.data:
            collected_state.update(event.data)
    return  collected_state

def get_from_user(prompt):
    """Format user prompt"""
    return {"role": "user", "answer": prompt}
# Page config with beautiful theme
st.set_page_config(
    page_title="Main_Project",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CRITICAL: CSS must be loaded FIRST for immediate blue message styling
st.markdown("""
<style>
/* User message styling - MUST be defined early */
.user-message {
    background: #3b82f6 !important;
    color: white !important;
    padding: 0.75rem 1rem !important;
    border-radius: 7px !important;
    max-width: 95% !important;
}

/* Assistant message styling */
.assistant-message {
    background: #f1f5f9 !important;
    color: #334155 !important;
    padding: 0.75rem 1rem !important;
    border-radius: 12px !important;
    max-width: 85% !important;
}

.assistant-info {
    font-size: 0.875rem !important;
    color: #6b7280 !important;
    margin-bottom: 5px !important;
}
</style>
""", unsafe_allow_html=True)

# JavaScript for interactions
# st.markdown("""
# <script>
# function scrollToBottom() {
#     setTimeout(function() {
#         const mainContainer = document.querySelector('.main-container');
#         if (mainContainer) {
#             mainContainer.scrollTop = mainContainer.scrollHeight;
#         }
#         window.scrollTo(0, document.body.scrollHeight);
#     }, 100);
# }

# function toggleCode(header) {
#     const codeBlock = header.nextElementSibling;
#     const toggleText = header.querySelector('.toggle-text');
    
#     if (codeBlock.style.display === 'none') {
#         codeBlock.style.display = 'block';
#         toggleText.textanswer = 'Click to collapse';
#     } else {
#         codeBlock.style.display = 'none';
#         toggleText.textanswer = 'Click to expand';
#     }
# }
# </script>
# """, unsafe_allow_html=True)

# FORCE reload environment variables
load_dotenv(override=True)



# Model order is decided by this
models = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b": "qwen/qwen3-32b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "llama4 maverik":"meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama3.3": "llama-3.3-70b-versatile",
    "deepseek-R1": "deepseek-r1-distill-llama-70b",
}

self_path = os.path.dirname(os.path.abspath(__file__))
print(self_path)

# Initialize session ID for this session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Filter available models
available_models = []
model_names = list(models.keys())
groq_models = []

for model_name in model_names:
    if "gemini" not in model_name:
        groq_models.append(model_name)

available_models.extend(groq_models)


# Set GPT-OSS-120B as default if available
default_index = 0
if "gpt-oss-120b" in available_models:
    default_index = available_models.index("gpt-oss-120b")
elif "deepseek-R1" in available_models:
    default_index = available_models.index("deepseek-R1")

# Compact header - everything perfectly aligned at same height
st.markdown("""
<style>
.header-container {
    display: flex; 
    align-items: center; 
     justify-content: center;
    gap: 12px;
    border-bottom: 1px solid #e5e7eb;
}

.header-container img {
    height: 80px;
}

.header-container h1 {
    padding: 0.25rem 0;
    margin: 0; 
    font-size: 1.5rem; 
    font-weight: 700; 
    color: #2563eb;
}

/* 🔹 Responsive: On small screens stack vertically */
@media (max-width: 768px) {
    .header-container {
        flex-direction: column;
        text-align: center;
        gap: 0;
        padding: 0 0 0.40rem;
    }
    .header-container img {
        height: 60px;
    }
    .header-container h1 {
        padding: 0 0;
        font-size: 1.25rem;
    }
}
</style>
<div class="header-container">
    <div style="display: flex; flex-direction: column; line-height: 1.2;">
        <h1>Chat</h1>
        <span>AI Air Quality Analysisr</span>
    </div>
</div>
""", unsafe_allow_html=True)




# Clean sidebar  
with st.sidebar:
    # Model selector at top of sidebar for easy access
    model_name = st.selectbox(
        "🤖 AI Model:",
        available_models,
        index=default_index,
        help="Choose your AI model - easily accessible without scrolling!"
    )
    
    st.markdown("---")
    
    # Quick Queries Section
    st.markdown("### 💭 Quick Queries")
    
    # Load quick prompts with caching
    @st.cache_data
    def load_questions():
        questions = []
        questions_file = join(self_path, "questions.txt")
        if os.path.exists(questions_file):
            try:
                with open(questions_file, 'r', encoding='utf-8') as f:
                    answer = f.read()
                    questions = [q.strip() for q in answer.split("\n") if q.strip()]
            except Exception as e:
                questions = []
        return questions
    
    questions = load_questions()
    
    #
    
    # Quick query buttons in sidebar
    selected_prompt = None
    
    
    # Show all questions but in a scrollable format
    if len(questions) > 0:
        st.markdown("**Select a question to analyze:**")
        
        # Getting Started section with simple questions
        getting_started_questions = questions[:10]  # First 10 simple questions
        with st.expander("🚀 Getting Started - Simple Questions", expanded=True):
            for i, q in enumerate(getting_started_questions):
                if st.button(q, key=f"start_q_{i}", use_container_width=True, help=f"Analyze: {q}"):
                    selected_prompt = q
                    st.session_state.last_selected_prompt = q
        
        # Create expandable sections for better organization
        with st.expander("📊 NCAP Funding & Policy Analysis", expanded=False):
            for i, q in enumerate([q for q in questions if any(word in q.lower() for word in ['ncap', 'funding', 'investment', 'rupee'])]):
                if st.button(q, key=f"ncap_q_{i}", use_container_width=True, help=f"Analyze: {q}"):
                    selected_prompt = q
                    st.session_state.last_selected_prompt = q
        
        with st.expander("🌬️ Meteorology & Environmental Factors", expanded=False):
            for i, q in enumerate([q for q in questions if any(word in q.lower() for word in ['wind', 'temperature', 'humidity', 'rainfall', 'meteorological', 'monsoon', 'barometric'])]):
                if st.button(q, key=f"met_q_{i}", use_container_width=True, help=f"Analyze: {q}"):
                    selected_prompt = q
                    st.session_state.last_selected_prompt = q
        
        with st.expander("👥 Population & Demographics", expanded=False):
            for i, q in enumerate([q for q in questions if any(word in q.lower() for word in ['population', 'capita', 'density', 'exposure'])]):
                if st.button(q, key=f"pop_q_{i}", use_container_width=True, help=f"Analyze: {q}"):
                    selected_prompt = q
                    st.session_state.last_selected_prompt = q
        
        with st.expander("🏭 Multi-Pollutant Analysis", expanded=False):
            for i, q in enumerate([q for q in questions if any(word in q.lower() for word in ['ozone', 'no2', 'correlation', 'multi-pollutant', 'interaction'])]):
                if st.button(q, key=f"multi_q_{i}", use_container_width=True, help=f"Analyze: {q}"):
                    selected_prompt = q
                    st.session_state.last_selected_prompt = q
        
        with st.expander("📈 Other Analysis Questions", expanded=False):
            remaining_questions = [q for q in questions if not any(any(word in q.lower() for word in category) for category in [
                ['ncap', 'funding', 'investment', 'rupee'],
                ['wind', 'temperature', 'humidity', 'rainfall', 'meteorological', 'monsoon', 'barometric'],
                ['population', 'capita', 'density', 'exposure'],
                ['ozone', 'no2', 'correlation', 'multi-pollutant', 'interaction']
            ])]
            for i, q in enumerate(remaining_questions):
                if st.button(q, key=f"other_q_{i}", use_container_width=True, help=f"Analyze: {q}"):
                    selected_prompt = q
                    st.session_state.last_selected_prompt = q
    
    st.markdown("---")
    
    
    # Clear Chat Button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.responses = []
        st.session_state.processing = False
        st.session_state.session_id = str(uuid.uuid4())
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

# Initialize session state first
if "responses" not in st.session_state:
    st.session_state.responses = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())




def show_custom_response(response):
    """Custom response display function with improved styling"""
    role = response.get("role", "assistant")
    answer = response.get("answer", "")

    
    if role == "user":
        # User message with right alignment - CSS now loaded at top of file
        st.markdown(f"""
        <div style='display: flex;  justify-content: flex-end; margin: 1rem 0;'>
            <div class='user-message'>
                {answer}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        # Check if answer is an image filename - don't display the filename text
        is_image_path = isinstance(answer, str) and any(ext in answer for ext in ['.png', '.jpg', '.jpeg'])
        
        # Check if answer is a pandas DataFrame
        import pandas as pd
        if isinstance(answer, list) and all(isinstance(row, dict) for row in answer):
            answer = pd.DataFrame(answer)
        is_dataframe = isinstance(answer, pd.DataFrame)
        
        # Check for errors first and display them with special styling
        error = response.get("error")
        timestamp = response.get("timestamp", "")
        timestamp_display = f" • {timestamp}" if timestamp else ""
        
        if error:
            st.markdown(f"""
            <div style='display: flex;  justify-content: flex-start; margin: 1rem 0;'>
                <div class='assistant-message'>
                    <div class='assistant-info'>Chat{timestamp_display}</div>
                    <div class='error-message'>
                        ⚠️ <strong>Error:</strong> {error}
                        <br><br>
                        <em>💡 Try rephrasing your question or being more specific about what you'd like to analyze.</em>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Assistant message with left alignment - reduced margins
        elif not is_image_path and not is_dataframe:
            st.markdown(f"""
            <div style='display: flex;  justify-content: flex-start; margin: 1rem 0;'>
                <div class='assistant-message'>
                    <div class='assistant-info'>Chatbot{timestamp_display}</div>
                    {answer if isinstance(answer, str) else str(answer)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif is_dataframe:
            # Display DataFrame with nice formatting
            st.markdown(f"""
            <div style='display: flex;  justify-content: flex-start; margin: 1rem 0;'>
                <div class='assistant-message'>
                    <div class='assistant-info'>Chatbot{timestamp_display}</div>
                    Here are the results:
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add context info for dataframes
            st.markdown("""
            <div class='context-info'>
                💡 This table is interactive - click column headers to sort, or scroll to view all data.
            </div>
            """, unsafe_allow_html=True)
            
            # Display dataframe with built-in download functionality
            st.dataframe(
                answer, 
                use_container_width=True,
                hide_index=True,
                column_config=None
            )
        
        # Show generated code with Streamlit expander
        if response.get("generated_code"):
            with st.expander("📋 View Generated Code", expanded=False):
                st.code(response["generated_code"], language="python")
        
        # Check if this is a plot response (plots are now displayed directly via st.pyplot)
        is_plot_response = isinstance(answer, str) and "Plot displayed successfully" in answer
        
        # Try to display image if answer is a file path (for backward compatibility)
        try:
            if isinstance(answer, str) and (answer.endswith('.png') or answer.endswith('.jpg')):
                answer = os.path.join("..\Backend", answer)
                if os.path.exists(answer):
                    # Display image with better styling and reasonable width
                    st.markdown("""
                    <div style='margin: 1rem 0; display: flex;  justify-content: center;'>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(answer, width=1080, caption="Generated Visualization")
                    return {"is_image": True}
            # Also handle case where answer shows filename but we want to show image
            elif isinstance(answer, str) and any(ext in answer for ext in ['.png', '.jpg']):
                # Extract potential filename from answer
                import re
                filename_match = re.search(r'([^/\\]+\.(?:png|jpg|jpeg))', answer)
                if filename_match:
                    filename = filename_match.group(1)
                    answer = os.path.join("Backend", answer)
                    if os.path.exists(filename):
                        st.markdown("""
                        <div style='margin: 1rem 0; display: flex;  justify-content: center;'>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(filename, width=1080, caption="Generated Visualization")
                        return {"is_image": True}
        except:
            pass
            
        return {"is_image": False}


# Chat history
# Display chat history
for response_id, response in enumerate(st.session_state.responses):
    status = show_custom_response(response)
    

# Chat input with better guidance
prompt = st.chat_input("💬 Ask about air quality trends, pollution analysis, or city comparisons...", key="main_chat")

# Handle selected prompt from quick prompts
if selected_prompt:
    prompt = selected_prompt

# Handle follow-up prompts from quick action buttons
if st.session_state.get("follow_up_prompt") and not st.session_state.get("processing"):
    prompt = st.session_state.follow_up_prompt
    st.session_state.follow_up_prompt = None  # Clear the follow-up prompt

# Handle new queries
if prompt and not st.session_state.get("processing"):
  

    if prompt:
        # Add user input to chat history
        user_response = get_from_user(prompt)
        st.session_state.responses.append(user_response)
        
        # Set processing state
        st.session_state.processing = True
        st.session_state.current_model = model_name
        st.session_state.current_question = prompt
        
        # Rerun to show processing indicator
        st.rerun()

# Process the question if we're in processing state
if st.session_state.get("processing"):
    # Enhanced processing indicator like Claude Code
    st.markdown("""
    <div style='padding: 1rem; text-align: center; background: #f8fafc; border-radius: 8px; margin: 1rem 0;'>
        <div style='display: flex; align-items: center;  justify-content: center; gap: 0.5rem; color: #475569;'>
            <div style='font-weight: 500;'>🤖 Processing with """ + str(st.session_state.get('current_model', 'Unknown')) + """</div>
            <div class='dots' style='display: inline-flex; gap: 2px;'>
                <div class='dot' style='width: 4px; height: 4px; background: #3b82f6; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out;'></div>
                <div class='dot' style='width: 4px; height: 4px; background: #3b82f6; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out; animation-delay: 0.16s;'></div>
                <div class='dot' style='width: 4px; height: 4px; background: #3b82f6; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out; animation-delay: 0.32s;'></div>
            </div>
        </div>
        <div style='font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;'>Analyzing data and generating response...</div>
    </div>
    <style>
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    prompt = st.session_state.get("current_question")
    model_name = st.session_state.get("current_model")
    
    try:
        response =  asyncio.run(ask_question(quest=prompt, model=model_name))
        if not isinstance(response, dict):
            response = {
                "role": "assistant",
                "answer": "Error: Invalid response format",
                "generated_code": "",
                "executed_code": "",
                "last_prompt": prompt,
                "error": "Invalid response format",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        
        response.setdefault("role", "assistant")
        response.setdefault("answer", "No answer generated")
        response.setdefault("generated_code", "")
        response.setdefault("executed_code", "")
        response.setdefault("last_prompt", prompt)
        response.setdefault("error", None)
        response.setdefault("timestamp", datetime.now().strftime("%H:%M"))
        
    except Exception as e:
        response = {
            "role": "assistant",
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "generated_code": "",
            "executed_code": "",
            "last_prompt": prompt,
            "error": str(e),
            "timestamp": datetime.now().strftime("%H:%M")
        }

    st.session_state.responses.append(response)
    st.session_state["last_prompt"] = prompt
    st.session_state["last_model_name"] = model_name
    st.session_state.processing = False
    
    # Clear processing state
    if "current_model" in st.session_state:
        del st.session_state.current_model
    if "current_question" in st.session_state:
        del st.session_state.current_question
    
    st.rerun()

# Close chat container
st.markdown("</div>", unsafe_allow_html=True)

# Make chat messages scrollable
st.markdown("""
<style>
/* ✅ Make chat messages scrollable without breaking layout */
[data-testid="stVerticalBlock"] > div {
    max-height: 75vh !important;   /* limit height to keep things visible */
    overflow-y: auto !important;   /* add scroll bar for long chats */
    padding-right: 10px !important;
}

/* Smooth scrolling for better UX */
[data-testid="stVerticalBlock"] > div::-webkit-scrollbar {
    width: 8px;
}
[data-testid="stVerticalBlock"] > div::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 10px;
}
[data-testid="stVerticalBlock"] > div::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# Minimal auto-scroll - only scroll when processing
if st.session_state.get("processing"):
    st.markdown("<script>scrollToBottom();</script>", unsafe_allow_html=True)

    # Dataset Info Section (matching mockup)
    st.markdown("### Dataset Info")
    st.markdown("""
    <div style='background: #f1f5f9; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;'>
        <h4 style='margin: 0 0 0.5rem 0; color: #1e293b; font-size: 0.9rem;'>PM2.5 Air Quality Data</h4>
        <p style='margin: 0; font-size: 0.75rem; color: #475569;'><strong>Time Range:</strong> 2022 - 2023</p>
        <p style='margin: 0; font-size: 0.75rem; color: #475569;'><strong>Locations:</strong> 300+ cities across India</p>
        <p style='margin: 0; font-size: 0.75rem; color: #475569;'><strong>Records:</strong> 100,000+ measurements</p>
    </div>
    """, unsafe_allow_html=True)
    

# streamlit adds each markdown's div, so its better to keep this in the last
# Custom CSS for beautiful styling
st.markdown("""
<style>
/* Clean app background */
.stApp {
    background-color: #ffffff;
    color: #212529;
    font-family: 'Segoe UI', sans-serif;
}

/* Reduce main container padding */
.main .block-container {
    padding-top: 0px;
    padding-bottom: 3rem;
    max-width: 100%;
}

/* Remove excessive spacing */
.element-container {
    margin-bottom: 0.5rem !important;
}

/* Fix sidebar spacing */
[data-testid="stSidebar"] .element-container {
    margin-bottom: 0.25rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #dee2e6;
    padding: 1rem;
}

/* Optimize sidebar scrolling */
[data-testid="stSidebar"] > div:first-child {
    height: 100vh;
    overflow-y: auto;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 6px;
}

[data-testid="stSidebar"]::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}

/* Main title */
.main-title {
    text-align: center;
    color: #343a40;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #6c757d;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}

/* Instructions */
.instructions {
    background-color: #f1f3f5;
    border-left: 4px solid #0d6efd;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-radius: 6px;
    color: #495057;
    text-align: left;
}

/* Quick prompt buttons */
.quick-prompt-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 10px;
    border: 1px solid #dee2e6;
}

.quick-prompt-btn {
    background-color: #0d6efd;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
}

.quick-prompt-btn:hover {
    background-color: #0b5ed7;
    transform: translateY(-2px);
}

/* User message styling */
.user-message {
    background: #3b82f6;
    color: white;
    padding: 0.75rem 1rem;
    border-radius: 7px;
    max-width: 95%;
}

.user-info {
    font-size: 0.875rem;
    opacity: 0.9;
    margin-bottom: 3px;
}

/* Assistant message styling */
.assistant-message {
    background: #f1f5f9;
    color: #334155;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    max-width: 85%;
}

.assistant-info {
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 5px;
}

/* Processing indicator */
.processing-indicator {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #333;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    margin-left: 0;
    margin-right: auto;
    max-width: 70%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* Feedback box */
.feedback-section {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

/* Success and error messages */
.success-message {
    background-color: #d1e7dd;
    color: #0f5132;
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid #badbcc;
}

.error-message {
    background-color: #f8d7da;
    color: #842029;
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid #f5c2c7;
}

/* Chat input styling - Fixed alignment */
# .stChatInput {
#     border-radius: 12px !important;
#     border: 2px solid #e5e7eb !important;
#     background: #ffffff !important;
#     padding: 0.75rem 1rem !important;
#     font-size: 1rem !important;
#     width: 100% !important;
#     max-width: 70% !important;
#     margin: 0 !important;
#     box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
#     transition: all 0.2s ease !important;
# }

# .stChatInput:focus {
#     border-color: #3b82f6 !important;
#     box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
#     outline: none !important;
# }

/* Chat input container */
.stChatInput > div {
    padding: 0 !important;
    margin: 0 !important;
}

/* Chat input text area */
# .stChatInput textarea {
#     border: none !important;
#     background: transparent !important;
#     padding: 0 !important;
#     margin: 0 !important;
#     font-size: 1rem !important;
#     line-height: 1.5 !important;
#     resize: none !important;
#     outline: none !important;
# }

/* Chat input placeholder */
# .stChatInput textarea::placeholder {
#     color: #9ca3af !important;
#     font-style: normal !important;
# }

.st-emotion-cache-f4ro0r {
    align-items = center;
}

/* Fix the main chat input container alignment */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0.5rem !important;
    left: 6rem !important;
    right: 0 !important;
    background: #ffffff !important;
    width: 65% !important;  
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1) !important;
}

/* Adjust main answer to account for fixed chat input */
.main .block-container {
    padding-bottom: 100px !important;
}

/* Chat input button styling */
[data-testid="stChatInput"] button {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease !important;
}

[data-testid="stChatInput"] button:hover {
    background: #2563eb !important;
}

/* Textarea inside chat input */
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    border: 2px solid #3b82f6 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    color: #111 !important;

    width: 100% !important;   /* fill the parent container */
    box-sizing: border-box !important;
}

/* Ensure proper spacing from sidebar */
@media (min-width: 768px) {
    [data-testid="stChatInput"] {
        margin-left: 21rem !important; /* Account for sidebar width */
    }
}

/* Code container styling */
.code-container {
    margin: 1rem 0;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    background: white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.code-header {
    display: flex;
     justify-content: space-between;
    align-items: center;
    padding: 0.875rem 1.25rem;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-bottom: 1px solid #e2e8f0;
    cursor: pointer;
    transition: all 0.2s ease;
    border-radius: 12px 12px 0 0;
}

.code-header:hover {
    background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
}

.code-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.code-title:before {
    content: "⚡";
    font-size: 0.8rem;
}

.toggle-text {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
}

.code-block {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
    padding: 1.5rem;
    font-family: 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 0.875rem;
    overflow-x: auto;
    line-height: 1.6;
    border-radius: 0 0 12px 12px;
}

.answer-container {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.answer-text {
    font-size: 1.125rem;
    color: #1e293b;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.answer-highlight {
    background: #fef3c7;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    font-weight: 600;
    color: #92400e;
}

.context-info {
    background: #f1f5f9;
    border-left: 4px solid #3b82f6;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    font-size: 0.875rem;
    color: #475569;
}

/* Hide default menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Auto scroll */
.main-container {
    height: 70vh;
    overflow-y: auto;
}

</style>

""", unsafe_allow_html=True)