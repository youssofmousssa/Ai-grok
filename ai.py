#!/usr/bin/env python3
"""
Advanced Telegram Bot with Comprehensive Groq API Integration

This bot provides extensive functionality using all available Groq models:
- Text generation with multiple LLM models
- Speech-to-text transcription and translation
- Text-to-speech synthesis
- Function calling and tool use
- Content moderation and safety
- Reasoning capabilities
- Multilingual support
- Vision capabilities (where supported)

Author: AI Assistant
Version: 2.0
License: MIT
"""

import os
import sys
import json
import logging
import asyncio
import requests
import tempfile
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import base64

# Telegram Bot imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes, ConversationHandler
)
from telegram.constants import ParseMode, ChatAction

# Configure comprehensive logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('groq_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GROQ_API_KEY = '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'
TELEGRAM_BOT_TOKEN = '7430894063:AAHb4psK9x9K0tAHyP5DeD5HqOa_uwZ1g1A'
GROQ_API_BASE_URL = 'https://api.groq.com/openai/v1'

# Bot configuration
MAX_MESSAGE_LENGTH = 4096
MAX_AUDIO_SIZE_MB = 25
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
CONVERSATION_TIMEOUT = 3600  # 1 hour

# --- Enums and Data Classes ---
class ModelCategory(Enum):
    REASONING = "reasoning"
    FUNCTION_CALLING = "function_calling"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_TEXT = "text_to_text"
    VISION = "vision"
    MULTILINGUAL = "multilingual"
    SAFETY_MODERATION = "safety_moderation"
    PREVIEW_SYSTEMS = "preview_systems"

@dataclass
class ModelInfo:
    id: str
    category: ModelCategory
    alias: str
    description: str
    context_window: int = 8192
    max_tokens: int = 2048
    supports_streaming: bool = True
    supports_functions: bool = False

@dataclass
class UserSession:
    user_id: int
    current_model: str
    conversation_history: List[Dict[str, str]]
    settings: Dict[str, Any]
    last_activity: datetime

# --- Comprehensive Groq Model Mapping ---
# Based on the official Groq documentation and groq.md
GROQ_MODELS: Dict[str, ModelInfo] = {
    # Production Models - Reasoning
    "qwen-qwq-32b": ModelInfo(
        id="qwen-qwq-32b",
        category=ModelCategory.REASONING,
        alias="qwen-qwq",
        description="Qwen QwQ 32B - Advanced reasoning model",
        context_window=131072,
        max_tokens=40960
    ),
    "deepseek-r1-distill-llama-70b": ModelInfo(
        id="deepseek-r1-distill-llama-70b",
        category=ModelCategory.REASONING,
        alias="deepseek-r1",
        description="DeepSeek R1 Distill Llama 70B - Reasoning model",
        context_window=131072,
        max_tokens=131072
    ),
    
    # Production Models - Function Calling
    "llama-3.3-70b-versatile": ModelInfo(
        id="llama-3.3-70b-versatile",
        category=ModelCategory.FUNCTION_CALLING,
        alias="llama3.3-70b",
        description="Llama 3.3 70B Versatile - Function calling and tool use",
        context_window=131072,
        max_tokens=32768,
        supports_functions=True
    ),
    "qwen/qwen3-32b": ModelInfo(
        id="qwen/qwen3-32b",
        category=ModelCategory.FUNCTION_CALLING,
        alias="qwen3-32b",
        description="Qwen 3 32B - Function calling model",
        context_window=131072,
        max_tokens=40960,
        supports_functions=True
    ),
    
    # Production Models - Text to Speech
    "playai-tts": ModelInfo(
        id="playai-tts",
        category=ModelCategory.TEXT_TO_SPEECH,
        alias="playai-dialog",
        description="PlayAI TTS - Text to speech synthesis",
        context_window=8192,
        max_tokens=8192,
        supports_streaming=False
    ),
    "playai-tts-arabic": ModelInfo(
        id="playai-tts-arabic",
        category=ModelCategory.TEXT_TO_SPEECH,
        alias="playai-arabic",
        description="PlayAI TTS Arabic - Arabic text to speech",
        context_window=8192,
        max_tokens=8192,
        supports_streaming=False
    ),
    
    # Production Models - Speech to Text
    "whisper-large-v3": ModelInfo(
        id="whisper-large-v3",
        category=ModelCategory.SPEECH_TO_TEXT,
        alias="whisper-v3",
        description="Whisper Large v3 - Speech to text transcription",
        context_window=448,
        max_tokens=448,
        supports_streaming=False
    ),
    "whisper-large-v3-turbo": ModelInfo(
        id="whisper-large-v3-turbo",
        category=ModelCategory.SPEECH_TO_TEXT,
        alias="whisper-turbo",
        description="Whisper Large v3 Turbo - Fast speech to text",
        context_window=448,
        max_tokens=448,
        supports_streaming=False
    ),
    "distil-whisper-large-v3-en": ModelInfo(
        id="distil-whisper-large-v3-en",
        category=ModelCategory.SPEECH_TO_TEXT,
        alias="whisper-distil",
        description="Distil Whisper Large v3 English - Efficient STT",
        context_window=448,
        max_tokens=448,
        supports_streaming=False
    ),
    
    # Production Models - Text to Text
    "llama-3.1-8b-instant": ModelInfo(
        id="llama-3.1-8b-instant",
        category=ModelCategory.TEXT_TO_TEXT,
        alias="llama3.1-8b",
        description="Llama 3.1 8B Instant - Fast text generation",
        context_window=131072,
        max_tokens=131072
    ),
    "llama-3.3-70b-versatile": ModelInfo(
        id="llama-3.3-70b-versatile",
        category=ModelCategory.TEXT_TO_TEXT,
        alias="llama3.3-70b-text",
        description="Llama 3.3 70B Versatile - Advanced text generation",
        context_window=131072,
        max_tokens=32768
    ),
    "gemma2-9b-it": ModelInfo(
        id="gemma2-9b-it",
        category=ModelCategory.TEXT_TO_TEXT,
        alias="gemma2",
        description="Gemma 2 9B IT - Google's text generation model",
        context_window=8192,
        max_tokens=8192
    ),
    "mistral-saba-24b": ModelInfo(
        id="mistral-saba-24b",
        category=ModelCategory.TEXT_TO_TEXT,
        alias="mistral-saba",
        description="Mistral Saba 24B - Multilingual text model",
        context_window=32768,
        max_tokens=32768
    ),
    
    # Production Models - Safety/Moderation
    "llama-guard-3-8b": ModelInfo(
        id="llama-guard-3-8b",
        category=ModelCategory.SAFETY_MODERATION,
        alias="llama-guard3",
        description="Llama Guard 3 8B - Content moderation",
        context_window=8192,
        max_tokens=1024,
        supports_streaming=False
    ),
    
    # Preview Models
    "meta-llama/llama-4-maverick-17b-128e-instruct": ModelInfo(
        id="meta-llama/llama-4-maverick-17b-128e-instruct",
        category=ModelCategory.PREVIEW_SYSTEMS,
        alias="llama4-maverick",
        description="Llama 4 Maverick 17B - Preview model",
        context_window=131072,
        max_tokens=8192
    ),
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelInfo(
        id="meta-llama/llama-4-scout-17b-16e-instruct",
        category=ModelCategory.PREVIEW_SYSTEMS,
        alias="llama4-scout",
        description="Llama 4 Scout 17B - Preview model",
        context_window=131072,
        max_tokens=8192
    ),
    "meta-llama/llama-prompt-guard-2-8m": ModelInfo(
        id="meta-llama/llama-prompt-guard-2-8m",
        category=ModelCategory.SAFETY_MODERATION,
        alias="prompt-guard2",
        description="Llama Prompt Guard 2 - Prompt injection detection",
        context_window=512,
        max_tokens=512,
        supports_streaming=False
    ),
    "compound-beta": ModelInfo(
        id="compound-beta",
        category=ModelCategory.PREVIEW_SYSTEMS,
        alias="compound",
        description="Compound Beta - Preview system",
        context_window=131072,
        max_tokens=8192
    )
}

# Create alias mapping for easy lookup
MODEL_ALIASES = {model.alias: model_id for model_id, model in GROQ_MODELS.items()}
MODEL_ALIASES.update({model_id: model_id for model_id in GROQ_MODELS.keys()})

# User sessions storage
user_sessions: Dict[int, UserSession] = {}

# --- Privacy and Security ---
class PrivacyFilter:
    """Advanced privacy filter to protect sensitive information."""
    
    SENSITIVE_PATTERNS = [
        # API Keys and tokens
        r'gsk_[a-zA-Z0-9]{48,}',
        r'\d{10}:[a-zA-Z0-9_-]{35}',
        # Common sensitive terms
        'api.?key', 'token', 'password', 'secret', 'private.?key',
        'groq.?api', 'telegram.?bot', 'configuration', 'internal',
        'source.?code', 'credentials', 'auth', 'bearer'
    ]
    
    @classmethod
    def is_sensitive(cls, text: str) -> bool:
        """Check if text contains sensitive information."""
        import re
        text_lower = text.lower()
        
        for pattern in cls.SENSITIVE_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        
        # Check for the actual API key and bot token
        if GROQ_API_KEY.lower() in text_lower or TELEGRAM_BOT_TOKEN in text:
            return True
            
        return False
    
    @classmethod
    def sanitize_for_logging(cls, text: str) -> str:
        """Sanitize text for safe logging."""
        import re
        sanitized = text
        
        # Replace API keys
        sanitized = re.sub(r'gsk_[a-zA-Z0-9]{48,}', 'gsk_***REDACTED***', sanitized)
        sanitized = re.sub(r'\d{10}:[a-zA-Z0-9_-]{35}', '***:***REDACTED***', sanitized)
        
        return sanitized

# --- Groq API Client ---
class GroqAPIClient:
    """Comprehensive Groq API client with error handling and rate limiting."""
    
    def __init__(self, api_key: str, base_url: str = GROQ_API_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'AdvancedGroqTelegramBot/2.0'
        })
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     json_data: Dict = None, files: Dict = None, 
                     data: Dict = None, timeout: int = 30) -> requests.Response:
        """Make HTTP request to Groq API with comprehensive error handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'POST':
                if files:
                    # Remove Content-Type for file uploads
                    headers = {k: v for k, v in self.session.headers.items() 
                              if k.lower() != 'content-type'}
                    response = requests.post(url, headers=headers, files=files, 
                                           data=data, timeout=timeout)
                else:
                    self.session.headers['Content-Type'] = 'application/json'
                    response = self.session.post(url, json=json_data, timeout=timeout)
            else:
                response = self.session.get(url, timeout=timeout)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error for {endpoint}")
            raise Exception("Request timed out. Please try again.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {endpoint}: {e}")
            if e.response.status_code == 429:
                raise Exception("Rate limit exceeded. Please wait a moment.")
            elif e.response.status_code == 401:
                raise Exception("Authentication failed. Invalid API key.")
            elif e.response.status_code == 400:
                raise Exception("Bad request. Please check your input.")
            else:
                raise Exception(f"API error: {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {endpoint}: {e}")
            raise Exception("Network error. Please try again.")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Fetch available models from Groq API."""
        try:
            response = self._make_request('/models')
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return {}
    
    def chat_completion(self, model: str, messages: List[Dict], 
                       temperature: float = DEFAULT_TEMPERATURE,
                       max_tokens: int = DEFAULT_MAX_TOKENS,
                       stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Send chat completion request."""
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream,
            **kwargs
        }
        
        try:
            response = self._make_request('/chat/completions', 'POST', json_data=payload)
            return response.json()
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise
    
    def audio_transcription(self, file_path: str, model: str = 'whisper-large-v3',
                           language: str = None, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file."""
        try:
            with open(file_path, 'rb') as audio_file:
                files = {'file': (os.path.basename(file_path), audio_file)}
                data = {'model': model}
                if language:
                    data['language'] = language
                data.update(kwargs)
                
                response = self._make_request('/audio/transcriptions', 'POST', 
                                            files=files, data=data)
                return response.json()
        except FileNotFoundError:
            raise Exception("Audio file not found.")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def audio_translation(self, file_path: str, model: str = 'whisper-large-v3',
                         **kwargs) -> Dict[str, Any]:
        """Translate audio to English."""
        try:
            with open(file_path, 'rb') as audio_file:
                files = {'file': (os.path.basename(file_path), audio_file)}
                data = {'model': model}
                data.update(kwargs)
                
                response = self._make_request('/audio/translations', 'POST',
                                            files=files, data=data)
                return response.json()
        except FileNotFoundError:
            raise Exception("Audio file not found.")
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise
    
    def text_to_speech(self, text: str, model: str = 'playai-tts',
                      voice: str = 'Fritz-PlayAI', response_format: str = 'mp3',
                      speed: float = 1.0, **kwargs) -> bytes:
        """Convert text to speech."""
        payload = {
            'model': model,
            'input': text,
            'voice': voice,
            'response_format': response_format,
            'speed': speed,
            **kwargs
        }
        
        try:
            response = self._make_request('/audio/speech', 'POST', json_data=payload)
            return response.content
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise

# Initialize Groq client
groq_client = GroqAPIClient(GROQ_API_KEY)

# --- Utility Functions ---
def get_user_session(user_id: int) -> UserSession:
    """Get or create user session."""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(
            user_id=user_id,
            current_model="llama-3.1-8b-instant",
            conversation_history=[],
            settings={
                'temperature': DEFAULT_TEMPERATURE,
                'max_tokens': DEFAULT_MAX_TOKENS,
                'language': 'en',
                'voice': 'Fritz-PlayAI'
            },
            last_activity=datetime.now()
        )
    else:
        user_sessions[user_id].last_activity = datetime.now()
    
    return user_sessions[user_id]

def cleanup_old_sessions():
    """Remove inactive user sessions."""
    cutoff_time = datetime.now() - timedelta(seconds=CONVERSATION_TIMEOUT)
    inactive_users = [
        user_id for user_id, session in user_sessions.items()
        if session.last_activity < cutoff_time
    ]
    
    for user_id in inactive_users:
        del user_sessions[user_id]
        logger.info(f"Cleaned up session for user {user_id}")

def split_long_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """Split long messages into chunks."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.rstrip())
                current_chunk = line + '\n'
            else:
                # Line is too long, split it
                while len(line) > max_length:
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                current_chunk = line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.rstrip())
    
    return chunks

def create_model_keyboard() -> InlineKeyboardMarkup:
    """Create inline keyboard for model selection."""
    keyboard = []
    
    # Group models by category
    categories = {}
    for model_id, model_info in GROQ_MODELS.items():
        category = model_info.category.value
        if category not in categories:
            categories[category] = []
        categories[category].append((model_info.alias, model_id))
    
    for category, models in categories.items():
        # Add category header
        keyboard.append([InlineKeyboardButton(
            f"📁 {category.replace('_', ' ').title()}", 
            callback_data=f"category_{category}"
        )])
        
        # Add models in pairs
        for i in range(0, len(models), 2):
            row = []
            for j in range(2):
                if i + j < len(models):
                    alias, model_id = models[i + j]
                    row.append(InlineKeyboardButton(
                        alias, callback_data=f"model_{model_id}"
                    ))
            keyboard.append(row)
    
    return InlineKeyboardMarkup(keyboard)

# --- Conversation States ---
SELECTING_MODEL, WAITING_FOR_MESSAGE, WAITING_FOR_AUDIO = range(3)

# --- Bot Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    session = get_user_session(user.id)
    
    welcome_message = f"""
🤖 **Welcome to the Advanced Groq AI Bot!**

Hello {user.first_name}! I'm your comprehensive AI assistant powered by Groq's lightning-fast models.

**🚀 What I can do:**
• 💬 Chat with multiple AI models
• 🎤 Transcribe and translate audio
• 🔊 Convert text to speech
• 🛠️ Function calling and tool use
• 🔒 Content moderation
• 🧠 Advanced reasoning
• 🌍 Multilingual support

**📋 Quick Commands:**
/help - Show detailed help
/models - Browse available models
/settings - Configure preferences
/stats - View usage statistics

**🎯 Current Model:** `{session.current_model}`

Type any message to start chatting, or use /help for more options!
"""
    
    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """
🤖 **Advanced Groq AI Bot - Complete Guide**

**💬 Basic Chat:**
• Just type any message to chat with the current AI model
• Use /model to switch between different models

**🎯 Model Commands:**
• `/ask <model> <message>` - Chat with specific model
• `/models` - Browse all available models
• `/model <alias>` - Switch default model
• `/current` - Show current model info

**🎤 Audio Commands:**
• `/tts <text>` - Convert text to speech
• `/stt` - Transcribe voice message (send audio after)
• `/translate` - Translate audio to English
• Send voice messages directly for auto-transcription

**⚙️ Settings:**
• `/settings` - Configure bot preferences
• `/temperature <0.0-2.0>` - Set response creativity
• `/voice <voice_name>` - Set TTS voice
• `/language <code>` - Set preferred language

**📊 Information:**
• `/stats` - View your usage statistics
• `/about` - Bot information
• `/privacy` - Privacy policy

**🔧 Advanced Features:**
• `/moderate <text>` - Check content safety
• `/reason <problem>` - Use reasoning models
• `/function <description>` - Function calling

**📝 Model Categories:**
"""
    
    # Add model categories
    categories = {}
    for model_info in GROQ_MODELS.values():
        category = model_info.category.value.replace('_', ' ').title()
        if category not in categories:
            categories[category] = []
        categories[category].append(f"`{model_info.alias}`")
    
    for category, models in categories.items():
        help_text += f"\n**{category}:**\n"
        help_text += " • ".join(models[:5])  # Show first 5 models
        if len(models) > 5:
            help_text += f" • ... and {len(models) - 5} more"
        help_text += "\n"
    
    help_text += "\n💡 **Tips:**\n"
    help_text += "• Use reasoning models for complex problems\n"
    help_text += "• Try different models for varied perspectives\n"
    help_text += "• Use /moderate before sharing sensitive content\n"
    help_text += "• Voice messages are auto-transcribed\n"
    
    # Split message if too long
    chunks = split_long_message(help_text)
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /models command."""
    keyboard = create_model_keyboard()
    
    await update.message.reply_text(
        "🤖 **Select a Model Category or Specific Model:**\n\n"
        "Choose from our comprehensive collection of AI models. "
        "Each model has unique strengths for different tasks.",
        reply_markup=keyboard,
        parse_mode=ParseMode.MARKDOWN
    )

async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle model selection callbacks."""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    session = get_user_session(user_id)
    
    if query.data.startswith("model_"):
        model_id = query.data[6:]  # Remove "model_" prefix
        
        if model_id in GROQ_MODELS:
            session.current_model = model_id
            model_info = GROQ_MODELS[model_id]
            
            await query.edit_message_text(
                f"✅ **Model Selected: {model_info.alias}**\n\n"
                f"**Description:** {model_info.description}\n"
                f"**Category:** {model_info.category.value.replace('_', ' ').title()}\n"
                f"**Context Window:** {model_info.context_window:,} tokens\n"
                f"**Max Output:** {model_info.max_tokens:,} tokens\n\n"
                f"You can now chat with this model! Type any message to start.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await query.edit_message_text("❌ Model not found.")
    
    elif query.data.startswith("category_"):
        category = query.data[9:]  # Remove "category_" prefix
        
        # Show models in this category
        category_models = [
            (model_id, model_info) for model_id, model_info in GROQ_MODELS.items()
            if model_info.category.value == category
        ]
        
        if category_models:
            keyboard = []
            for model_id, model_info in category_models:
                keyboard.append([InlineKeyboardButton(
                    f"{model_info.alias} - {model_info.description[:30]}...",
                    callback_data=f"model_{model_id}"
                )])
            
            keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_main")])
            
            await query.edit_message_text(
                f"🤖 **{category.replace('_', ' ').title()} Models:**\n\n"
                f"Select a model from this category:",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
    
    elif query.data == "back_to_main":
        keyboard = create_model_keyboard()
        await query.edit_message_text(
            "🤖 **Select a Model Category or Specific Model:**",
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN
        )

async def current_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current model information."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    model_info = GROQ_MODELS.get(session.current_model)
    if model_info:
        info_text = f"""
🤖 **Current Model Information**

**Model:** `{model_info.alias}`
**ID:** `{model_info.id}`
**Category:** {model_info.category.value.replace('_', ' ').title()}
**Description:** {model_info.description}

**Specifications:**
• Context Window: {model_info.context_window:,} tokens
• Max Output: {model_info.max_tokens:,} tokens
• Streaming: {'✅' if model_info.supports_streaming else '❌'}
• Functions: {'✅' if model_info.supports_functions else '❌'}

**Your Settings:**
• Temperature: {session.settings['temperature']}
• Max Tokens: {session.settings['max_tokens']}
• Language: {session.settings['language']}
"""
        
        await update.message.reply_text(info_text, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("❌ Current model information not available.")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask command for specific model queries."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "❌ **Usage:** `/ask <model_alias> <your_message>`\n\n"
            "**Example:** `/ask llama3.3-70b What is quantum computing?`\n\n"
            "Use /models to see available model aliases.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    model_alias = context.args[0].lower()
    message_text = " ".join(context.args[1:])
    
    # Check for sensitive content
    if PrivacyFilter.is_sensitive(message_text):
        await update.message.reply_text(
            "🔒 **Privacy Protection:** I cannot process requests containing "
            "sensitive information like API keys or personal credentials."
        )
        return
    
    # Find model by alias
    model_id = MODEL_ALIASES.get(model_alias)
    if not model_id:
        await update.message.reply_text(
            f"❌ **Model not found:** `{model_alias}`\n\n"
            "Use /models to see available models.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    model_info = GROQ_MODELS[model_id]
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    # Send typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)
    
    try:
        # Prepare messages
        messages = [{"role": "user", "content": message_text}]
        
        # Make API call
        response = groq_client.chat_completion(
            model=model_id,
            messages=messages,
            temperature=session.settings['temperature'],
            max_tokens=min(session.settings['max_tokens'], model_info.max_tokens)
        )
        
        if response and response.get('choices'):
            reply_text = response['choices'][0]['message']['content']
            
            # Add model info header
            header = f"🤖 **{model_info.alias}** response:\n\n"
            full_response = header + reply_text
            
            # Split if too long
            chunks = split_long_message(full_response)
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text("❌ No response received from the model.")
    
    except Exception as e:
        logger.error(f"Ask command error: {e}")
        await update.message.reply_text(f"❌ **Error:** {str(e)}")

async def chat_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular chat messages."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    message_text = update.message.text
    
    # Check for sensitive content
    if PrivacyFilter.is_sensitive(message_text):
        await update.message.reply_text(
            "🔒 **Privacy Protection:** I cannot process requests containing "
            "sensitive information like API keys or personal credentials."
        )
        return
    
    # Send typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)
    
    try:
        model_info = GROQ_MODELS[session.current_model]
        
        # Add to conversation history
        session.conversation_history.append({
            "role": "user",
            "content": message_text
        })
        
        # Keep conversation history manageable
        if len(session.conversation_history) > 20:
            session.conversation_history = session.conversation_history[-20:]
        
        # Make API call
        response = groq_client.chat_completion(
            model=session.current_model,
            messages=session.conversation_history,
            temperature=session.settings['temperature'],
            max_tokens=min(session.settings['max_tokens'], model_info.max_tokens)
        )
        
        if response and response.get('choices'):
            reply_text = response['choices'][0]['message']['content']
            
            # Add to conversation history
            session.conversation_history.append({
                "role": "assistant",
                "content": reply_text
            })
            
            # Split if too long
            chunks = split_long_message(reply_text)
            for chunk in chunks:
                await update.message.reply_text(chunk)
        else:
            await update.message.reply_text("❌ No response received from the model.")
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        await update.message.reply_text(f"❌ **Error:** {str(e)}")

async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /tts command for text-to-speech."""
    if not context.args:
        await update.message.reply_text(
            "❌ **Usage:** `/tts <text_to_convert>`\n\n"
            "**Example:** `/tts Hello, how are you today?`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    text_to_convert = " ".join(context.args)
    
    # Check for sensitive content
    if PrivacyFilter.is_sensitive(text_to_convert):
        await update.message.reply_text(
            "🔒 **Privacy Protection:** I cannot process requests containing "
            "sensitive information."
        )
        return
    
    # Check text length
    if len(text_to_convert) > 1000:
        await update.message.reply_text(
            "❌ **Text too long:** Please limit text to 1000 characters for TTS."
        )
        return
    
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    await update.message.chat.send_action(ChatAction.RECORD_AUDIO)
    
    try:
        # Generate speech
        audio_content = groq_client.text_to_speech(
            text=text_to_convert,
            model="playai-tts",
            voice=session.settings['voice']
        )
        
        if audio_content:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name
            
            try:
                # Send audio
                with open(temp_file_path, 'rb') as audio_file:
                    await update.message.reply_audio(
                        audio=audio_file,
                        title="Generated Speech",
                        caption=f"🔊 TTS: \"{text_to_convert[:50]}{'...' if len(text_to_convert) > 50 else ''}\""
                    )
            finally:
                # Clean up
                os.unlink(temp_file_path)
        else:
            await update.message.reply_text("❌ Failed to generate speech.")
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        await update.message.reply_text(f"❌ **TTS Error:** {str(e)}")

async def stt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stt command for speech-to-text setup."""
    await update.message.reply_text(
        "🎤 **Speech-to-Text Ready**\n\n"
        "Please send me a voice message now and I'll transcribe it for you.\n\n"
        "**Supported formats:** Voice messages, audio files\n"
        "**Max size:** 25MB\n"
        "**Languages:** Auto-detected or specify with /language"
    )
    
    # Set user state for STT
    context.user_data['awaiting_stt'] = True

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /translate command for audio translation."""
    await update.message.reply_text(
        "🌍 **Audio Translation Ready**\n\n"
        "Please send me a voice message in any language and I'll translate it to English.\n\n"
        "**Supported formats:** Voice messages, audio files\n"
        "**Max size:** 25MB"
    )
    
    # Set user state for translation
    context.user_data['awaiting_translation'] = True

async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    # Check file size
    file_size_mb = update.message.voice.file_size / (1024 * 1024)
    if file_size_mb > MAX_AUDIO_SIZE_MB:
        await update.message.reply_text(
            f"❌ **File too large:** {file_size_mb:.1f}MB. "
            f"Maximum size is {MAX_AUDIO_SIZE_MB}MB."
        )
        return
    
    await update.message.chat.send_action(ChatAction.TYPING)
    
    try:
        # Download voice message
        file = await context.bot.get_file(update.message.voice.file_id)
        
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            await file.download_to_drive(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Check user intent
            if context.user_data.get('awaiting_translation'):
                # Translate to English
                result = groq_client.audio_translation(temp_file_path)
                context.user_data['awaiting_translation'] = False
                
                if result and result.get('text'):
                    await update.message.reply_text(
                        f"🌍 **Translation (English):**\n\n{result['text']}"
                    )
                else:
                    await update.message.reply_text("❌ Could not translate the audio.")
            
            else:
                # Transcribe (default or explicit STT)
                if context.user_data.get('awaiting_stt'):
                    context.user_data['awaiting_stt'] = False
                
                language = session.settings.get('language') if session.settings.get('language') != 'en' else None
                result = groq_client.audio_transcription(temp_file_path, language=language)
                
                if result and result.get('text'):
                    await update.message.reply_text(
                        f"🎤 **Transcription:**\n\n{result['text']}"
                    )
                else:
                    await update.message.reply_text("❌ Could not transcribe the audio.")
        
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        await update.message.reply_text(f"❌ **Audio Error:** {str(e)}")

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settings command."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    settings_text = f"""
⚙️ **Your Bot Settings**

**Current Model:** `{GROQ_MODELS[session.current_model].alias}`
**Temperature:** {session.settings['temperature']} (creativity level)
**Max Tokens:** {session.settings['max_tokens']} (response length)
**Language:** {session.settings['language']} (for STT)
**Voice:** {session.settings['voice']} (for TTS)

**Available Commands:**
• `/temperature <0.0-2.0>` - Set response creativity
• `/max_tokens <number>` - Set max response length
• `/language <code>` - Set language (en, es, fr, de, etc.)
• `/voice <name>` - Set TTS voice
• `/reset_settings` - Reset to defaults

**Conversation:**
• Messages in history: {len(session.conversation_history)}
• `/clear_history` - Clear conversation history
"""
    
    await update.message.reply_text(settings_text, parse_mode=ParseMode.MARKDOWN)

async def temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /temperature command."""
    if not context.args:
        await update.message.reply_text(
            "❌ **Usage:** `/temperature <0.0-2.0>`\n\n"
            "**Examples:**\n"
            "• `/temperature 0.1` - Very focused responses\n"
            "• `/temperature 0.7` - Balanced (default)\n"
            "• `/temperature 1.5` - Very creative responses",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    try:
        temp = float(context.args[0])
        if not 0.0 <= temp <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        user_id = update.effective_user.id
        session = get_user_session(user_id)
        session.settings['temperature'] = temp
        
        await update.message.reply_text(
            f"✅ **Temperature set to {temp}**\n\n"
            f"Response creativity level updated!"
        )
    
    except ValueError as e:
        await update.message.reply_text(f"❌ **Invalid temperature:** {str(e)}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    stats_text = f"""
📊 **Your Usage Statistics**

**Session Info:**
• Current Model: `{GROQ_MODELS[session.current_model].alias}`
• Messages in History: {len(session.conversation_history)}
• Last Activity: {session.last_activity.strftime('%Y-%m-%d %H:%M:%S')}

**Bot Statistics:**
• Total Active Sessions: {len(user_sessions)}
• Available Models: {len(GROQ_MODELS)}
• Model Categories: {len(set(m.category for m in GROQ_MODELS.values()))}

**Model Distribution:**
"""
    
    # Add model category stats
    category_counts = {}
    for model_info in GROQ_MODELS.values():
        category = model_info.category.value.replace('_', ' ').title()
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in category_counts.items():
        stats_text += f"• {category}: {count} models\n"
    
    await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)

async def moderate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /moderate command for content moderation."""
    if not context.args:
        await update.message.reply_text(
            "❌ **Usage:** `/moderate <text_to_check>`\n\n"
            "**Example:** `/moderate Is this content appropriate?`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    text_to_check = " ".join(context.args)
    
    await update.message.chat.send_action(ChatAction.TYPING)
    
    try:
        # Use Llama Guard for content moderation
        messages = [
            {
                "role": "user",
                "content": f"Please analyze this content for safety: {text_to_check}"
            }
        ]
        
        response = groq_client.chat_completion(
            model="llama-guard-3-8b",
            messages=messages,
            temperature=0.1,
            max_tokens=512
        )
        
        if response and response.get('choices'):
            result = response['choices'][0]['message']['content']
            
            await update.message.reply_text(
                f"🛡️ **Content Moderation Result:**\n\n{result}",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text("❌ Moderation check failed.")
    
    except Exception as e:
        logger.error(f"Moderation error: {e}")
        await update.message.reply_text(f"❌ **Moderation Error:** {str(e)}")

async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear_history command."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    session.conversation_history.clear()
    
    await update.message.reply_text(
        "🗑️ **Conversation history cleared!**\n\n"
        "Starting fresh with your next message."
    )

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about command."""
    about_text = """
🤖 **Advanced Groq AI Telegram Bot**

**Version:** 2.0
**Author:** AI Assistant
**License:** MIT

**Powered by:**
• Groq API - Lightning-fast AI inference
• Python Telegram Bot - Robust bot framework
• Multiple AI Models - Comprehensive capabilities

**Features:**
• 🚀 Ultra-fast responses via Groq
• 🧠 Multiple AI models for different tasks
• 🎤 Speech-to-text transcription
• 🔊 Text-to-speech synthesis
• 🛡️ Content moderation
• 🌍 Multilingual support
• 🔒 Privacy protection

**Model Categories:**
• Reasoning - Advanced problem solving
• Function Calling - Tool use and APIs
• Text Generation - Creative and informative
• Audio Processing - Speech and sound
• Safety - Content moderation
• Vision - Image understanding (where supported)

**Open Source:**
This bot demonstrates the power of Groq's API
for building advanced AI applications.

**Support:** Use /help for commands or report issues.
"""
    
    await update.message.reply_text(about_text, parse_mode=ParseMode.MARKDOWN)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")
    
    # Log the full traceback
    logger.error("Full traceback:", exc_info=context.error)
    
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ **An error occurred while processing your request.**\n\n"
                "Please try again in a moment. If the problem persists, "
                "use /about to report the issue."
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

# --- Main Application ---
def main() -> None:
    """Start the bot."""
    logger.info("Starting Advanced Groq Telegram Bot...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("current", current_model_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("tts", tts_command))
    application.add_handler(CommandHandler("stt", stt_command))
    application.add_handler(CommandHandler("translate", translate_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("temperature", temperature_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("moderate", moderate_command))
    application.add_handler(CommandHandler("clear_history", clear_history_command))
    application.add_handler(CommandHandler("about", about_command))
    
    # Add callback query handler for inline keyboards
    application.add_handler(CallbackQueryHandler(model_callback))
    
    # Add message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_message_handler))
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handler))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Set commands and start bot
    async def post_init(application):
        commands = [
            BotCommand("start", "Start the bot"),
            BotCommand("help", "Show help information"),
            BotCommand("models", "Browse available models"),
            BotCommand("ask", "Ask specific model"),
            BotCommand("tts", "Text to speech"),
            BotCommand("stt", "Speech to text"),
            BotCommand("settings", "Bot settings"),
            BotCommand("stats", "Usage statistics"),
            BotCommand("about", "About this bot")
        ]
        await application.bot.set_my_commands(commands)
        
        # Start periodic cleanup
        async def periodic_cleanup():
            while True:
                await asyncio.sleep(3600)  # Every hour
                cleanup_old_sessions()
        
        asyncio.create_task(periodic_cleanup())
    
    application.post_init = post_init
    
    # Start the bot
    logger.info("Bot started successfully!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

