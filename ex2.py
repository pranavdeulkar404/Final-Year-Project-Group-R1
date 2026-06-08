import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import threading
import queue
import time
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import sqlite3
from streamlit_option_menu import option_menu  # Import the option menu

# Configuration
GOOGLE_API_KEY = "AIzaSyCGXiRNKcdrgYZDtOzAv7Oefsa7Mj7gWWQ"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Database Configuration (SQLite)
DATABASE_NAME = "user_database.db"

# Global State Management
class AppState:
    def __init__(self):
        self.chat_history = []
        self.cap = None
        self.video_placeholder = None
        self.emotion_queue = queue.Queue()
        self.voice_prompt = ""
        self.voice_input_active = False
        self.current_mode = "text"
        self.model_name = "gemini-2.0-flash"
        self.text_response_enabled = True
        self.last_emotion = None
        self.emotion_history = []
        self.user = None  # To store logged-in user info
        self.page = "home" # Added page state variable for navigation

state = st.session_state.setdefault('state', AppState())

# Database Helper Functions
def get_db_connection():
    """Gets a database connection."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

def create_tables():
    """Creates the user table if it doesn't exist."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user'
                )
            """)
            conn.commit()
        except Exception as e:
            st.error(f"Error creating tables: {e}")
        finally:
            conn.close()

create_tables()  # Ensure tables exist on startup


def get_user(username):
    """Retrieves a user from the database by username."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            return user
        except Exception as e:
            st.error(f"Error retrieving user: {e}")
            return None
        finally:
            conn.close()
    return None

def add_user(username, password, role='user'):
    """Adds a new user to the database."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error adding user: {e}")
            return False
        finally:
            conn.close()
    return False

def delete_user(user_id):
    """Deletes a user from the database by ID."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error deleting user: {e}")
            return False
        finally:
            conn.close()
    return False

def get_all_users():
    """Retrieves all users from the database."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()
            return users
        except Exception as e:
            st.error(f"Error retrieving users: {e}")
            return []
        finally:
            conn.close()
    return None

# Authentication Functions
def login(username, password):
    """Validates user credentials."""
    user = get_user(username)
    if user and user['password'] == password:  # Simple password check (INSECURE - use hashing in real app)
        return user
    return None

def signup(username, password):
    """Registers a new user."""
    if get_user(username):
        st.error("Username already exists")
        return False
    return add_user(username, password)

# Text-to-Speech Function
def speak(text):
    """Enhanced text-to-speech with interrupt capability"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()

# Unified Gemini Response Function with Emotion Integration
def get_gemini_response(prompt, emotion=None):
    """Response generator that incorporates emotional context"""
    try:
        model = genai.GenerativeModel(state.model_name)
        context = [
            "System: You are an emotionally-aware AI assistant. Respond appropriately to the user's emotional state when detected. Also take into account the sentiment from the text input to provide emotionally equivalent output.",
            f"Mode: {state.current_mode.upper()}",
            *[msg for msg in state.chat_history[-6:] if msg],
        ]

        if emotion:
            context.insert(1, f"System: User's current emotional state appears to be {emotion.upper()}.")
            state.last_emotion = emotion

        full_context = "\n".join(context)
        response = model.generate_content(full_context)

        if response and response.text:
            stored_prompt = f"User: {prompt}"
            if emotion:
                stored_prompt += f" [Emotion: {emotion}]"
            state.chat_history.extend([stored_prompt, f"Assistant: {response.text}"])
            return response.text

        st.error("Empty response from model")
        return None

    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Enhanced Video Processing with Emotion Detection
def video_processor():
    """Real-time face and emotion analysis"""
    state.cap = cv2.VideoCapture(0)
    if not state.cap.isOpened():
        st.error("Cannot access webcam")
        return

    try:
        while state.cap.isOpened():
            ret, frame = state.cap.read()
            if not ret:
                st.warning("Can't receive frame")
                break

            try:
                # Analyze frame for emotions
                results = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(results, list):
                    for face in results:
                        if (
                            isinstance(face, dict) and
                            'region' in face and isinstance(face['region'], dict) and
                            'x' in face['region'] and 'y' in face['region'] and
                            'w' in face['region'] and 'h' in face['region'] and
                            'dominant_emotion' in face
                        ):
                            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            emotion = face['dominant_emotion']
                            cv2.putText(
                                frame, f"{emotion}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2
                            )
                            # Update emotion queue and history
                            try:
                                state.emotion_queue.put_nowait(emotion)
                                state.emotion_history.append((time.time(), emotion))
                            except queue.Full:
                                pass
                        else:
                            st.error(f"Invalid face data structure: {face}")
                            print(f"Invalid face data structure: {face}")
                            continue

                    # Display video feed
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    state.video_placeholder.image(frame_rgb, channels="RGB")

            except Exception as e:
                st.error(f"Error processing frame: {e}")
                print(f"Error processing frame: {e}")
                continue  # Continue to the next frame

            time.sleep(0.1)

    finally:
        if state.cap:
            state.cap.release()

# Voice Processing Function
def voice_processor():
    """Handles voice input with emotion context"""
    recognizer = sr.Recognizer()
    while state.voice_input_active:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                state.voice_prompt = text
                state.voice_input_active = False
            except sr.RequestError as e:
                st.error(f"Error with the request to Google Speech Recognition service: {e}")
                print(f"Error with the request to Google Speech Recognition service: {e}")
                state.voice_input_active = False
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand audio")
                print("Google Speech Recognition could not understand audio")
                state.voice_input_active = False
            except Exception as e:
                st.error(f"Error in voice_processor: {e}")
                print(f"Error in voice_processor: {e}")
                state.voice_input_active = False
        time.sleep(0.1)



# Sidebar Configuration - Removed to integrate into main page
# def setup_sidebar():
#     """App configuration panel"""
#     with st.sidebar:
#         st.header("Settings")
#         state.model_name = st.selectbox(
#             "AI Model",
#             ["gemini-2.0-flash", "gemini-1.5-pro"],
#             index=0
#         )
#         state.text_response_enabled = st.checkbox("Voice Responses", True)
#
#         if st.button("Clear History"):
#             state.chat_history.clear()
#             state.emotion_history.clear()
#             st.rerun()
#
#         if state.last_emotion:
#             st.write(f"Last detected emotion: {state.last_emotion}")

# Input Processing Functions
def process_text_input(prompt):
    if prompt:
        response = get_gemini_response(prompt)
        if response and state.text_response_enabled:
            speak(response)

def process_voice_input():
    if state.voice_prompt:
        emotion = None
        try:
            emotion = state.emotion_queue.get_nowait()
        except queue.Empty:
            pass
        response = get_gemini_response(state.voice_prompt, emotion)
        if response:
            speak(response)
        state.voice_prompt = ""



def process_multimodal_input():
    if state.voice_prompt:
        emotion = None
        try:
            emotion = state.emotion_queue.get_nowait()
        except queue.Empty:
            pass
        response = get_gemini_response(state.voice_prompt, emotion)
        if response:
            speak(response)
        state.voice_prompt = ""

# UI Components
def text_input():
    with st.form("text_form"):
        prompt = st.text_input("Message:", key="text_input")
        if st.form_submit_button("Send"):
            process_text_input(prompt)

def voice_input():
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎤 Start"):
            state.voice_input_active = True
            threading.Thread(target=voice_processor, daemon=True).start()
        if st.button("⏹️ Stop"):
            state.voice_input_active = False
    if state.voice_prompt:
        process_voice_input()



def multimodal_input():
    state.video_placeholder = st.empty()
    threading.Thread(target=video_processor, daemon=True).start()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Start Recording", key="multi_rec"):
            state.voice_input_active = True
            threading.Thread(target=voice_processor, daemon=True).start()
    with col2:
        if st.button("⏹️ Stop All", key="multi_stop"):
            state.voice_input_active = False
            if state.cap:
                state.cap.release()

    if state.voice_prompt:
        process_multimodal_input()



# Main Chat Interface - Renamed to avoid conflict
def emotion_ai_page():
    st.title("🤖 Emotion-Aware AI Assistant")

    # Mode selection
    state.current_mode = st.radio(
        "Mode",
        ["Text", "Voice", "Multimodal"],
        horizontal=True,
        index=0
    ).lower()

    # Chat display
    chat_container = st.container(height=400)
    with chat_container:
        for msg in state.chat_history[-10:]:
            if msg.startswith("User:"):
                st.markdown(f"**{msg}**")
            else:
                st.markdown(msg)

    # Input handling
    if state.current_mode == "text":
        text_input()
    elif state.current_mode == "voice":
        voice_input()
    else:
        multimodal_input()

# Page Functions
def home_page():
    st.title("Home Page")
    st.markdown("""
    Welcome to the Emotion-Aware AI Assistant Web Application! This platform is designed to understand and respond to your emotions using advanced AI. Our application uses your webcam and microphone (with your permission) to analyze your facial expressions and voice patterns, allowing the AI to adapt its responses to your emotional state.

    Key Features

    * Emotion Detection: Real-time analysis of your emotions.
    * Adaptive Responses: AI responses that are tailored to your emotions.
    * Text, Voice, and Multimodal Input: Interact with the AI in the way that suits you best.
    * User Authentication: Secure login and signup functionality.
    * Admin Panel: User management for administrators.

    How to Get Started

    1.  Sign Up or Log In: Create an account or log in to access the application's features.
    2.  Explore the Modes: Choose between text, voice, or multimodal input to interact with the AI.
    3.  Interact with the AI: Start a conversation and see how the AI responds to your emotions.
    4.  Check out the About Us page to learn more about our team and mission.
    """)

def about_us_page():
    st.title("About Us")
    st.markdown("""
    ## About Us

    Welcome to the Emotion-Aware AI Assistant, a cutting-edge web application developed by a team of passionate researchers and engineers dedicated to creating more human-centered AI experiences. Our mission is to bridge the gap between humans and machines by enabling AI to understand and respond to human emotions in a nuanced and empathetic manner.

    ### Our Team

    Our team comprises experts in artificial intelligence, computer vision, natural language processing, and software development. We are driven by a shared vision of creating AI that can enhance human well-being, improve communication, and provide personalized assistance across various domains.

    ### Our Technology

    The Emotion-Aware AI Assistant leverages state-of-the-art technologies, including:

    * **Facial Expression Recognition:** We employ deep learning models, such as Convolutional Neural Networks (CNNs), to analyze facial expressions captured by your webcam. These models are trained on vast datasets of facial images to accurately识别 a wide range of emotions, including joy, sadness, anger, fear, surprise, and neutrality.
    * **Voice Analysis:** We utilize advanced audio processing techniques and machine learning algorithms to analyze vocal cues, such as tone, pitch, and speech rate, to infer emotional states from voice patterns.
    * **Natural Language Processing (NLP):** Our system incorporates powerful NLP models, including transformer networks, to understand the semantic meaning and emotional sentiment behind your text-based input. This allows the AI to respond not only to the literal content of your messages but also to the underlying emotions they convey.
    * **Large Language Models (LLMs):** The application is powered by Google's Gemini family of large language models.  These models are capable of generating highly coherent and contextually relevant responses.
    """)

def contact_us_page():
    st.title("Contact Us")
    st.markdown("""
    ## Contact Us

    Thank you for your interest in the Emotion-Aware AI Assistant. We value your feedback, questions, and suggestions. Please feel free to reach out to us using the information below.

    ### General Inquiries

    For general inquiries about the application, its features, or our team, please contact us at:

    * Email: info@emotionaiassistant.com

    ### Technical Support

    If you are experiencing technical issues or need assistance with using the application, please contact our support team:

    * Email: support@emotionaiassistant.com

    ### Feedback and Suggestions

    We are always striving to improve our application. If you have any feedback or suggestions, please share them with us at:

    * Email: feedback@emotionaiassistant.com

    ### Mailing Address

    Emotion-Aware AI Assistant Team
    [Your Company Address]
    [Your City, State, Zip Code]
    """)

def login_page():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            user = login(username, password)
            if user:
                state.user = user
                state.page = "emotion_ai"  # Go to the emotion AI page after login
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")

def signup_page():
    st.title("Sign Up")
    with st.form("signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Sign Up"):
            if password == confirm_password:
                if signup(username, password):
                    st.success("Account created successfully! Please log in.")
                    state.page = "login" # go to login page
                    st.rerun()
                else:
                    st.error("Failed to create account")
            else:
                st.error("Passwords do not match")

def admin_page():
    """Admin page with user management functionality."""
    if not state.user or state.user['role'] != 'user':
        st.error("You do not have permission to access this page.")
        return

    st.title("Admin Page")
    st.header("User Management")

    users = get_all_users()
    if users:
        st.dataframe(users)  # Display user data in a table

        # User deletion
        user_id_to_delete = st.number_input("Enter User ID to Delete:", min_value=1, step=1)
        if st.button("Delete User"):
            if delete_user(user_id_to_delete):
                st.success(f"User with ID {user_id_to_delete} deleted successfully.")
                st.rerun()  # Refresh the page to update the user list
            else:
                st.error("Failed to delete user.")

        # User creation
        st.subheader("Add New User")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        # new_role = st.selectbox("Role", ["user", "admin"], index=0)
        new_role = st.selectbox("Role", ["user"], index=0)
        if st.button("Add User"):
            if add_user(new_username, new_password, new_role):
                st.success(f"User {new_username} added successfully.")
                st.rerun()  # Refresh the page
            else:
                st.error("Failed to add user.")
    else:
        st.write("No users found.")



# Main App Execution
if __name__ == "__main__":
    if not hasattr(st.session_state, 'initialized'):
        st.session_state.initialized = True
        state = AppState()

    st.set_page_config(
        page_title="Emotion AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    #setup_sidebar() # Removed to integrate into main page.
    #render_chat_interface() # Removed to use page state.

    # Sidebar and Navigation
    with st.sidebar:
        st.header("Navigation")
        menu_options = ["Home", "Login", "Sign Up", "About Us", "Contact Us", "Admin"] # Added Admin to nav
        if state.user:
            menu_options.append("Emotion AI")
            if state.user['role'] == 'admin':
                menu_options.append("Admin")
        selected_page = option_menu(
            "Menu",
            menu_options,
            icons=['house', 'arrow-in-right', 'arrow-up', 'info-circle', 'envelope', 'gear'] + (['robot'] if state.user else []) + (['gear'] if state.user and state.user['role'] == 'admin' else []),
            default_index=0,
        )
        state.page = selected_page.lower().replace(" ", "_")

        st.header("Settings")
        state.model_name = st.selectbox(
            "AI Model",
            ["gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )
        state.text_response_enabled = st.checkbox("Voice Responses", True)

        if st.button("Clear History"):
            state.chat_history.clear()
            state.emotion_history.clear()
            st.rerun()

        if state.last_emotion:
            st.write(f"Last detected emotion: {state.last_emotion}")


    # Page Routing
    if state.page == "home":
        home_page()
    elif state.page == "about_us":
        about_us_page()
    elif state.page == "contact_us":
        contact_us_page()
    elif state.page == "login":
        login_page()
    elif state.page == "sign_up":
        signup_page()
    elif state.page == "emotion_ai":
        emotion_ai_page()
    elif state.page == "admin":
        admin_page()
    else:
        home_page() # default page

    # Cleanup
    if state.cap:
        state.cap.release()
    cv2.destroyAllWindows()
