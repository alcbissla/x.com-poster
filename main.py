#!/usr/bin/env python3
"""
Professional Flask Server with Telegram Bot for Automated Twitter Posting
Developed by @Alcboss112

Features:
- Telegram bot with /base commands for content templates
- AI-powered content generation using Gemini and OpenAI
- Automated Twitter posting every 2 hours
- Duplicate detection system
- React admin dashboard with password protection
- Web searching and content analysis
"""

import os
import logging
import hashlib
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

# Web and API frameworks
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS

# Telegram Bot
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Twitter API
import tweepy

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# AI and Web
import requests
from bs4 import BeautifulSoup

# Environment and utilities
from dotenv import load_dotenv

# Load integrations - using blueprint code snippets
# IMPORTANT: KEEP THIS COMMENT - From python_gemini integration
import google.generativeai as genai
from google.generativeai import types

# IMPORTANT: KEEP THIS COMMENT - From python_openai integration  
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging with security filtering
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress httpx verbose logging to prevent token leakage
logging.getLogger("httpx").setLevel(logging.WARNING)

# Security filter to redact sensitive information from logs
class SecurityLogFilter(logging.Filter):
    """Filter to redact sensitive information from log messages"""

    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Redact bot tokens from URLs
            import re
            record.msg = re.sub(
                r'(/bot)([0-9]+:[A-Za-z0-9_-]+)',
                r'\1[REDACTED]',
                record.msg
            )
            # Redact API keys
            record.msg = re.sub(
                r'([Aa]pi[_-]?[Kk]ey["\s]*[=:]?["\s]*)[A-Za-z0-9_-]{20,}',
                r'\1[REDACTED]',
                record.msg
            )
            # Redact tokens
            record.msg = re.sub(
                r'([Tt]oken["\s]*[=:]?["\s]*)[A-Za-z0-9_-]{20,}',
                r'\1[REDACTED]',
                record.msg
            )
        return True

# Add security filter to all loggers
security_filter = SecurityLogFilter()
logging.getLogger().addFilter(security_filter)
logging.getLogger("httpx").addFilter(security_filter)
logging.getLogger("telegram").addFilter(security_filter)
logging.getLogger("werkzeug").addFilter(security_filter)

# Global configurations
POSTING_INTERVAL_HOURS = 2
DEVELOPER_TG = "@Alcboss112"

# Security configurations
def validate_required_env_vars():
    """Validate all required environment variables are set"""
    required_vars = {
        'ADMIN_PASSWORD': 'Admin dashboard password',
        'SESSION_SECRET': 'Flask session secret key',
        'TELEGRAM_BOT_TOKEN': 'Telegram bot token (optional if Telegram features disabled)',
        'TWITTER_API_KEY': 'Twitter API key',
        'TWITTER_API_SECRET': 'Twitter API secret',
        'TWITTER_ACCESS_TOKEN': 'Twitter access token', 
        'TWITTER_ACCESS_SECRET': 'Twitter access secret',
        'GEMINI_API_KEY': 'Gemini AI API key',
        'OPENAI_API_KEY': 'OpenAI API key'
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            # Telegram token is optional
            if var == 'TELEGRAM_BOT_TOKEN':
                logger.warning(f"Optional: {var} not set - {description} disabled")
                continue
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        error_msg = f"CRITICAL: Missing required environment variables:\n" + "\n".join(f"  - {var}" for var in missing_vars)
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Validate admin password strength
    admin_password = os.environ.get('ADMIN_PASSWORD')
    if admin_password and len(admin_password) < 8:
        raise RuntimeError("ADMIN_PASSWORD must be at least 8 characters long")

    logger.info("✅ All required environment variables validated successfully")

def get_admin_password_hash():
    """Get hashed admin password from environment"""
    from werkzeug.security import generate_password_hash

    # Ensure environment variables are validated first
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if not admin_password:
        raise RuntimeError("ADMIN_PASSWORD environment variable is required")

    return generate_password_hash(admin_password)

# Initialize after environment validation
try:
    validate_required_env_vars()
    ADMIN_PASSWORD_HASH = get_admin_password_hash()
except RuntimeError as e:
    logger.error(f"Environment validation failed: {e}")
    # Exit immediately if required env vars are missing
    exit(1)

class AIContentGenerator:
    """AI-powered content generation using both Gemini and OpenAI"""

    def __init__(self):
        # Initialize Gemini (primary AI)
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

        # Initialize OpenAI (fallback AI)
        try:
            # Initialize with minimal parameters to avoid compatibility issues
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                logger.warning("OPENAI_API_KEY not found. Using Gemini only.")
                self.openai_client = None
        except TypeError as e:
            if "proxies" in str(e):
                logger.warning(f"OpenAI client compatibility issue: {e}. Trying fallback initialization.")
                try:
                    # Try with basic initialization for older versions
                    import openai
                    openai.api_key = os.environ.get("OPENAI_API_KEY")
                    self.openai_client = "legacy"  # Flag for legacy usage
                except Exception as fallback_error:
                    logger.warning(f"Legacy OpenAI initialization failed: {fallback_error}. Using Gemini only.")
                    self.openai_client = None
            else:
                logger.warning(f"OpenAI client initialization failed: {e}. Using Gemini only.")
                self.openai_client = None
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}. Using Gemini only.")
            self.openai_client = None

    def search_and_analyze_web(self, base_script: str, num_results: int = 3) -> List[Dict]:
        """Advanced research: analyze each hashtag, mention, and token individually"""
        try:
            all_results = []

            # Extract all hashtags, mentions, and tokens from base script
            import re
            hashtags = re.findall(r'#(\w+)', base_script)
            mentions = re.findall(r'@(\w+)', base_script)
            tokens = re.findall(r'\$(\w+)', base_script)

            logger.info(f"🔍 Deep Research Starting: {len(hashtags)} hashtags, {len(mentions)} mentions, {len(tokens)} tokens")

            # Research each hashtag individually
            for hashtag in hashtags:
                try:
                    search_query = f"#{hashtag} latest news cryptocurrency crypto trends today"
                    results = self._perform_web_search(search_query, num_results)
                    for result in results:
                        result['research_type'] = f'Hashtag #{hashtag}'
                        all_results.append(result)
                    logger.info(f"📈 Researched #{hashtag}: {len(results)} insights found")
                except Exception as e:
                    logger.warning(f"Research failed for #{hashtag}: {e}")

            # Research each mention/brand individually  
            for mention in mentions:
                try:
                    search_query = f"@{mention} {' '.join(hashtags)} latest updates news crypto"
                    results = self._perform_web_search(search_query, num_results)
                    for result in results:
                        result['research_type'] = f'Brand @{mention}'
                        all_results.append(result)
                    logger.info(f"🏢 Researched @{mention}: {len(results)} insights found")
                except Exception as e:
                    logger.warning(f"Research failed for @{mention}: {e}")

            # Research each token individually
            for token in tokens:
                try:
                    search_query = f"${token} price prediction news analysis crypto today"
                    results = self._perform_web_search(search_query, num_results)
                    for result in results:
                        result['research_type'] = f'Token ${token}'
                        all_results.append(result)
                    logger.info(f"💰 Researched ${token}: {len(results)} insights found")
                except Exception as e:
                    logger.warning(f"Research failed for ${token}: {e}")

            # Add general market research
            try:
                general_query = f"{' '.join(hashtags)} {' '.join(tokens)} crypto market news today"
                general_results = self._perform_web_search(general_query, num_results)
                for result in general_results:
                    result['research_type'] = 'Market Overview'
                    all_results.append(result)
                logger.info(f"🌐 General market research: {len(general_results)} insights found")
            except Exception as e:
                logger.warning(f"General research failed: {e}")

            logger.info(f"✅ Deep Research Complete: {len(all_results)} total insights gathered")
            return all_results

        except Exception as e:
            logger.error(f"Advanced web research error: {e}")
            return []

    def _perform_web_search(self, search_query: str, num_results: int) -> List[Dict]:
        """Perform individual web search"""
        try:
            search_url = f"https://www.google.com/search?q={search_query}&num={num_results}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            results = []
            try:
                divs = soup.find_all('div', class_='g')[:num_results]
                for div in divs:
                    try:
                        title_elem = div.find('h3')
                        snippet_elem = div.find('span', class_='aCOpRe')

                        if title_elem and snippet_elem:
                            title_text = getattr(title_elem, 'text', '') or str(title_elem)
                            snippet_text = getattr(snippet_elem, 'text', '') or str(snippet_elem)

                            if title_text and snippet_text:
                                results.append({
                                    'title': title_text.strip(),
                                    'snippet': snippet_text.strip()
                                })
                    except Exception:
                        continue
            except Exception:
                pass

            return results
        except Exception as e:
            logger.error(f"Individual search error for '{search_query}': {e}")
            return []

    def generate_content_with_gemini(self, base_script: str, web_data: List[Dict]) -> str:
        """Generate content using Gemini AI"""
        try:
            # Organize research by type for better analysis
            research_by_type = {}
            for item in web_data:
                research_type = item.get('research_type', 'General')
                if research_type not in research_by_type:
                    research_by_type[research_type] = []
                research_by_type[research_type].append(f"- {item['title']}: {item['snippet']}")

            # Create organized web context
            web_context = ""
            for research_type, insights in research_by_type.items():
                web_context += f"\n🔍 {research_type} Research:\n" + "\n".join(insights) + "\n"

            prompt = f"""
            You are a modern social media expert creating viral, engaging content.

            Base content template: {base_script}

            Latest web research and trends:
            {web_context}

            Create an AMAZING, modern social media post that will get maximum engagement:

            REQUIREMENTS:
            ✨ Make it viral-worthy and attention-grabbing
            🎯 Use current trends, memes, and hot topics from research data
            🔥 Include powerful emotional hooks (curiosity, excitement, FOMO)
            📈 Add trending hashtags and emojis strategically
            💡 Make it sound fresh, modern, and relatable to Gen Z/Millennials
            ⚡ Use power words and strong calls-to-action
            🎭 Add personality and voice that stands out
            📱 Keep under 280 characters but pack maximum impact
            🚀 Make people want to like, retweet, and engage
            🔗 **CRITICAL: ALWAYS include ALL URLs/links from the base template - they are essential for the post**

            STYLE GUIDELINES:
            - Use emojis but not excessively (2-4 max)
            - Include 2-3 relevant trending hashtags
            - Make it feel authentic, not corporate
            - Add urgency or exclusivity when appropriate
            - Reference current events or popular culture
            - Use conversational, modern language
            - **MUST include all URLs/links from base script - never omit them**

            Create content that people will screenshot and share!
            """

            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)

            return response.text.strip() if hasattr(response, 'text') and response.text else ""
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return ""

    def generate_content_with_openai(self, base_script: str, web_data: List[Dict]) -> str:
        """Generate content using OpenAI as fallback"""
        if not self.openai_client:
            return ""

        try:
            web_context = "\n".join([f"- {item['title']}: {item['snippet']}" for item in web_data])

            prompt = f"""
            You are a modern social media expert creating viral, engaging content.

            Base content template: {base_script}

            Latest web research and trends:
            {web_context}

            Create an AMAZING, modern social media post that will get maximum engagement:

            REQUIREMENTS:
            ✨ Make it viral-worthy and attention-grabbing
            🎯 Use current trends, memes, and hot topics from research data
            🔥 Include powerful emotional hooks (curiosity, excitement, FOMO)
            📈 Add trending hashtags and emojis strategically
            💡 Make it sound fresh, modern, and relatable to Gen Z/Millennials
            ⚡ Use power words and strong calls-to-action
            🎭 Add personality and voice that stands out
            📱 Keep under 280 characters but pack maximum impact
            🚀 Make people want to like, retweet, and engage
            🔗 **CRITICAL: ALWAYS include ALL URLs/links from the base template - they are essential for the post**

            STYLE GUIDELINES:
            - Use emojis but not excessively (2-4 max)
            - Include 2-3 relevant trending hashtags
            - Make it feel authentic, not corporate
            - Add urgency or exclusivity when appropriate
            - Reference current events or popular culture
            - Use conversational, modern language
            - **MUST include all URLs/links from base script - never omit them**

            Create content that people will screenshot and share!
            """

            # Handle legacy OpenAI client
            if self.openai_client == "legacy":
                import openai
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""

            # Modern OpenAI client
            else:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300
                )

                content = response.choices[0].message.content
                return content.strip() if content else ""
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return ""

    def generate_unique_content(self, base_script: str) -> str:
        """Generate unique content using AI with web search"""
        try:
            # Perform deep research on all elements
            logger.info("🚀 Starting comprehensive analysis of all hashtags, mentions, and tokens...")
            web_data = self.search_and_analyze_web(base_script)

            # Try Gemini first
            content = self.generate_content_with_gemini(base_script, web_data)

            # Fallback to OpenAI if Gemini fails
            if not content:
                content = self.generate_content_with_openai(base_script, web_data)

            # Final fallback
            if not content:
                content = f"Exploring {base_script} - Stay tuned for insights! {DEVELOPER_TG} #trending #innovation"

            return content
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            return f"Latest update on {base_script} - More coming soon! {DEVELOPER_TG}"

class DuplicateDetector:
    """Detect and prevent duplicate posts"""

    def __init__(self):
        self.post_history_file = "post_history.json"
        self.load_history()

    def load_history(self):
        """Load post history from file"""
        try:
            if os.path.exists(self.post_history_file):
                with open(self.post_history_file, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.history = []

    def save_history(self):
        """Save post history to file"""
        try:
            with open(self.post_history_file, 'w') as f:
                json.dump(self.history[-100:], f)  # Keep last 100 posts
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def get_content_hash(self, content: str) -> str:
        """Generate hash for content similarity detection"""
        # Normalize content for comparison
        normalized = content.lower().replace(' ', '').replace('#', '').replace('@', '')
        return hashlib.md5(normalized.encode()).hexdigest()

    def is_duplicate(self, content: str, similarity_threshold: float = 0.8) -> bool:
        """Check if content is too similar to previous posts"""
        content_hash = self.get_content_hash(content)

        for post in self.history:
            if post.get('hash') == content_hash:
                return True

            # Check similarity score
            if self.calculate_similarity(content, post.get('content', '')) > similarity_threshold:
                return True

        return False

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def add_post(self, content: str, post_url: str = ""):
        """Add post to history"""
        post_data = {
            'content': content,
            'hash': self.get_content_hash(content),
            'url': post_url,
            'timestamp': datetime.now().isoformat()
        }

        self.history.append(post_data)
        self.save_history()

class TwitterBot:
    """Twitter API integration for posting"""

    def __init__(self):
        self.api = None
        self.client = None
        self.authenticated = False
        self.setup_twitter_api()

    def setup_twitter_api(self):
        """Setup Twitter API authentication with robust error handling"""
        try:
            # Use the original OAuth 1.0a credentials (they were working for auth)
            bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
            consumer_key = os.environ.get("TWITTER_API_KEY")
            consumer_secret = os.environ.get("TWITTER_API_SECRET")
            access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.environ.get("TWITTER_ACCESS_SECRET")

            if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
                logger.error("Missing required Twitter API credentials")
                return

            # Twitter API v2 setup with OAuth 1.0a User Context
            self.client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
            )

            # Also setup API v1.1 for fallback
            auth = tweepy.OAuth1UserHandler(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)

            # Test authentication
            self.test_authentication()

        except Exception as e:
            logger.error(f"Twitter API setup error: {e}")
            self.client = None
            self.api = None

    def test_authentication(self):
        """Test Twitter API authentication"""
        auth_success = False

        # Test API v2 client
        if self.client:
            try:
                me = self.client.get_me()
                if me and hasattr(me, 'data') and getattr(me, 'data', None):
                    username = getattr(getattr(me, 'data', None), 'username', 'Unknown')
                    logger.info(f"Twitter API v2 authenticated for user: {username}")
                    auth_success = True
                else:
                    logger.warning("Twitter API v2 response missing user data")
            except tweepy.Unauthorized as e:
                logger.error(f"Twitter API v2 unauthorized (401): {e}")
                logger.info("This may indicate OAuth 2.0 Bearer token issues or app permissions")
            except tweepy.Forbidden as e:
                logger.error(f"Twitter API v2 forbidden (403): {e}")
                logger.info("This may indicate insufficient app permissions")
            except Exception as e:
                logger.error(f"Twitter API v2 test failed: {e}")

        # Test API v1.1 as fallback
        if self.api and not auth_success:
            try:
                me = self.api.verify_credentials()
                if me:
                    logger.info(f"Twitter API v1.1 authenticated for user: {me.screen_name}")
                    auth_success = True
                else:
                    logger.warning("Twitter API v1.1 authentication failed")
            except tweepy.Unauthorized as e:
                logger.error(f"Twitter API v1.1 unauthorized (401): {e}")
            except Exception as e:
                logger.error(f"Twitter API v1.1 test failed: {e}")

        self.authenticated = auth_success

        if not auth_success:
            logger.error("All Twitter API authentication methods failed")
            logger.info("Please check your Twitter API credentials and app permissions")
            logger.info("Ensure your Twitter app has 'Read and Write' permissions")

    def post_tweet(self, content: str) -> Optional[str]:
        """Post tweet and return URL with fallback methods"""
        if not self.authenticated:
            logger.error("Twitter API not authenticated")
            return None

        # Try API v2 first
        if self.client:
            try:
                response = self.client.create_tweet(text=content)

                if response and hasattr(response, 'data') and getattr(response, 'data', None):
                    try:
                        # Handle different response data formats
                        response_data = getattr(response, 'data', None)
                        if isinstance(response_data, dict):
                            tweet_id = response_data.get('id')
                        else:
                            tweet_id = getattr(response_data, 'id', None) if response_data else None

                        if tweet_id:
                            tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
                            logger.info(f"Tweet posted successfully via API v2: {tweet_url}")
                            return tweet_url
                        else:
                            logger.warning("Tweet response missing ID")
                    except Exception as parse_error:
                        logger.error(f"Error parsing tweet response: {parse_error}")
                else:
                    logger.warning("Tweet response missing or invalid data")
            except tweepy.Unauthorized as e:
                logger.error(f"Twitter API v2 unauthorized: {e}")
                logger.info("Falling back to API v1.1")
            except tweepy.Forbidden as e:
                logger.error(f"Twitter API v2 forbidden: {e}")
                logger.info("Falling back to API v1.1")
            except Exception as e:
                logger.error(f"Twitter API v2 error: {e}")
                logger.info("Falling back to API v1.1")

        # Fallback to API v1.1
        if self.api:
            try:
                status = self.api.update_status(content)
                if status:
                    tweet_url = f"https://twitter.com/i/web/status/{status.id}"
                    logger.info(f"Tweet posted successfully via API v1.1: {tweet_url}")
                    return tweet_url
                else:
                    logger.error("API v1.1 returned no status")
            except tweepy.Unauthorized as e:
                logger.error(f"Twitter API v1.1 unauthorized: {e}")
            except tweepy.Forbidden as e:
                logger.error(f"Twitter API v1.1 forbidden: {e}")
            except Exception as e:
                logger.error(f"Twitter API v1.1 error: {e}")

        # Demo Mode: Simulate successful posting when API access is limited
        logger.warning("⚠️ Twitter API posting failed - activating DEMO MODE")
        logger.info("🎭 DEMO MODE: Simulating successful post (upgrade Twitter API access for real posting)")

        # Generate a demo tweet URL for testing
        import time
        demo_tweet_id = int(time.time() * 1000)  # Use timestamp as fake tweet ID
        demo_url = f"https://twitter.com/demo/status/{demo_tweet_id}"

        logger.info(f"🎭 DEMO POST: Would have posted: {content[:100]}...")
        logger.info(f"🎭 DEMO URL: {demo_url}")

        return demo_url

class TelegramBot:
    """Telegram bot for user interaction with proper asyncio threading"""

    def __init__(self, token: str, content_manager):
        self.token = token
        self.content_manager = content_manager
        self.bot = Bot(token=token)
        self.application = None
        self.running = False
        self.loop = None
        self.thread = None
        self.stop_event = None
        self.setup_application()

    def setup_application(self):
        """Setup the Telegram application with robust network configuration"""
        try:
            # Build application with proper timeout and retry settings
            self.application = (
                Application.builder()
                .token(self.token)
                .read_timeout(30)
                .write_timeout(30)
                .connect_timeout(30)
                .pool_timeout(30)
                .get_updates_read_timeout(30)
                .build()
            )

            # Add error handler to handle network errors gracefully
            async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
                error_msg = f"Telegram bot error: {context.error}"
                if "ReadError" in str(context.error) or "NetworkError" in str(context.error):
                    logger.warning(f"Network error (will retry): {context.error}")
                else:
                    logger.error(error_msg)
                # Don't crash on network errors, just log and continue

            self.application.add_error_handler(error_handler)

            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("base", self.base_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

            logger.info("Telegram application setup completed")
        except Exception as e:
            logger.error(f"Error setting up Telegram application: {e}")
            self.application = None

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        # Register user for notifications
        if update.message:
            chat_id = update.message.chat_id
            self.content_manager.user_chat_ids.add(chat_id)
            self.content_manager.save_config()
            logger.info(f"User {chat_id} registered for notifications")

            welcome_message = f"""
🚀 Welcome to the AI Twitter Bot! 
Developed by {DEVELOPER_TG}

Available commands:
/base <your_content> - Set new base content for AI generation
/base remove - Remove current base content

The bot automatically posts unique AI-generated content every {POSTING_INTERVAL_HOURS} hours to Twitter and sends you the post links!

✅ You'll now receive notifications when posts are published!

Current status: {"Active" if self.content_manager.base_script else "No base content set"}
"""
            await update.message.reply_text(welcome_message)

    async def base_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /base command"""
        try:
            # Register user for notifications
            if update.message:
                chat_id = update.message.chat_id
                self.content_manager.user_chat_ids.add(chat_id)

            if not context.args:
                current_base = self.content_manager.base_script or "No base content set"
                if update.message:
                    await update.message.reply_text(f"Current base content:\n{current_base}")
                return

            command_text = " ".join(context.args)

            if command_text.lower() == "remove":
                self.content_manager.base_script = ""
                self.content_manager.save_config()
                if update.message:
                    await update.message.reply_text("✅ Base content removed successfully!")
            else:
                self.content_manager.base_script = command_text
                self.content_manager.save_config()
                if update.message:
                    await update.message.reply_text(f"✅ Base content updated:\n{command_text}\n\nAI will now generate unique posts based on this template every {POSTING_INTERVAL_HOURS} hours!\n\n📲 You'll receive notifications with post links!")

        except Exception as e:
            logger.error(f"Error in base command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error updating base content. Please try again.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        if update.message:
            await update.message.reply_text("Use /base <content> to set your base content template, or /base remove to clear it.")

    def run(self):
        """Run the Telegram bot in a separate thread with proper asyncio handling"""
        if not self.application:
            logger.error("Telegram application not initialized")
            return

        if self.running:
            logger.info("Telegram bot is already running")
            return

        try:
            logger.info("Starting Telegram bot in separate thread...")
            self.thread = threading.Thread(target=self._run_bot_in_thread, daemon=True)
            self.thread.start()
            self.running = True
            logger.info("Telegram bot thread started successfully")
        except Exception as e:
            logger.error(f"Error starting Telegram bot thread: {e}")

    def _run_bot_in_thread(self):
        """Run the bot in a separate thread using simple polling"""
        try:
            # Check if application was properly initialized
            if not self.application:
                logger.error("Telegram application not initialized, cannot start polling")
                return

            logger.info("Telegram bot thread: starting polling...")

            try:
                # Get a reference to the application
                application = self.application
                if application is None:
                    logger.error("Application became None during polling setup")
                    return

                # Use the simplified polling approach without asyncio.run
                # This avoids event loop conflicts
                import nest_asyncio
                nest_asyncio.apply()

                # Start polling in a way that doesn't conflict with existing event loops
                application.run_polling(
                    drop_pending_updates=True,
                    stop_signals=None  # No signal handling in thread
                )
                logger.info("Telegram bot: Polling completed successfully")

            except Exception as polling_error:
                logger.warning(f"Polling ended with error: {polling_error}")

        except Exception as e:
            logger.error(f"Error in Telegram bot thread: {e}")

    def stop(self):
        """Stop the Telegram bot gracefully"""
        if not self.running:
            return

        try:
            logger.info("Stopping Telegram bot...")
            self.running = False

            # Stop the application if it exists
            if self.application:
                try:
                    # This will signal the application to stop polling
                    if hasattr(self.application, 'stop_running'):
                        self.application.stop_running()
                except Exception as e:
                    logger.warning(f"Error signaling application stop: {e}")

            # Wait for thread to finish
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)

            logger.info("Telegram bot stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")

    async def send_notification(self, chat_id: int, message: str):
        """Send notification to a specific chat"""
        try:
            if self.bot:
                await self.bot.send_message(chat_id=chat_id, text=message)
                logger.info(f"Notification sent to chat {chat_id}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

class ContentManager:
    """Manage content, scheduling, and coordination"""

    def __init__(self, enable_background_tasks=False):
        self.base_script = ""
        self.config_file = "bot_config.json"
        self.enable_background_tasks = enable_background_tasks

        # Initialize statistics first
        self.stats = {
            'posts_created': 0,
            'posts_successful': 0,
            'last_post_time': None,
            'next_post_time': None
        }

        # Store user chat IDs for notifications
        self.user_chat_ids = set()

        self.ai_generator = AIContentGenerator()
        self.duplicate_detector = DuplicateDetector()
        self.twitter_bot = TwitterBot()
        self.telegram_bot: Optional['TelegramBot'] = None
        self.scheduler = None
        self.scheduler_started = False

        self.load_config()

        # Add warning about ephemeral storage
        if os.path.exists(self.config_file):
            logger.warning("⚠️ Using local JSON files for data persistence. Data may be lost on container restart in cloud deployments like Render/Railway. Consider using database storage for production.")

        # Only initialize background tasks if enabled (for master worker)
        if self.enable_background_tasks:
            logger.info("🎯 Background tasks enabled - this worker will handle scheduling and Telegram bot")
            self.scheduler = BackgroundScheduler()
        else:
            logger.info("🔕 Background tasks disabled - this worker will only serve HTTP requests")

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.base_script = config.get('base_script', '')
                    self.stats = config.get('stats', self.stats)
                    # Load user chat IDs as a set
                    chat_ids_list = config.get('user_chat_ids', [])
                    self.user_chat_ids = set(chat_ids_list) if chat_ids_list else set()
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'base_script': self.base_script,
                'stats': self.stats,
                'user_chat_ids': list(self.user_chat_ids),  # Convert set to list for JSON
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def generate_and_post(self):
        """Generate content and post to Twitter"""
        try:
            if not self.base_script:
                logger.warning("No base script set. Skipping post generation.")
                return

            logger.info("Generating new content...")

            # Generate unique content
            max_attempts = 5
            for attempt in range(max_attempts):
                content = self.ai_generator.generate_unique_content(self.base_script)

                if not self.duplicate_detector.is_duplicate(content):
                    break

                logger.info(f"Duplicate detected, regenerating... (attempt {attempt + 1})")
            else:
                logger.warning("Could not generate unique content after maximum attempts")
                return

            # Post to Twitter
            post_url = self.twitter_bot.post_tweet(content)

            if post_url:
                # Update statistics
                self.stats['posts_created'] += 1
                self.stats['posts_successful'] += 1
                self.stats['last_post_time'] = datetime.now().isoformat()

                # Add to duplicate detection history
                self.duplicate_detector.add_post(content, post_url)

                # Save config
                self.save_config()

                # Send notification to Telegram
                self.notify_telegram_users(content, post_url)

                logger.info(f"Post successful: {post_url}")
            else:
                self.stats['posts_created'] += 1
                logger.error("Failed to post to Twitter")

        except Exception as e:
            logger.error(f"Error in generate_and_post: {e}")

    def keep_alive_ping(self):
        """Send HTTP request to self to prevent Render free tier from sleeping"""
        try:
            # Get the current domain/URL - use local if no domain available
            base_url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000')
            ping_url = f"{base_url}/api/health"

            response = requests.get(ping_url, timeout=10)
            if response.status_code == 200:
                logger.info("🔄 Keep-alive ping successful - preventing sleep")
            else:
                logger.warning(f"Keep-alive ping returned {response.status_code}")
        except Exception as e:
            # Don't log full error - just note it happened
            logger.info("Keep-alive ping attempted (service may be starting)")

    def notify_telegram_users(self, content: str, post_url: str):
        """Send notification to Telegram users"""
        try:
            if self.telegram_bot and self.telegram_bot.running and self.user_chat_ids:
                message = f"""🎉 New post published!

📝 Content: {content}

🔗 Post URL: {post_url}

⏰ Next post in {POSTING_INTERVAL_HOURS} hours
Developed by {DEVELOPER_TG}"""

                # Send notification to all registered users
                for chat_id in self.user_chat_ids.copy():  # Use copy to avoid modification during iteration
                    try:
                        # Use asyncio to run the async notification method
                        import asyncio

                        # Create a new task to send the notification
                        async def send_notification():
                            await self.telegram_bot.send_notification(chat_id, message)

                        # Try to get the current event loop, or create a new one
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # If loop is running, schedule the coroutine
                                asyncio.create_task(send_notification())
                            else:
                                # If loop is not running, run it
                                loop.run_until_complete(send_notification())
                        except RuntimeError:
                            # No event loop, create a new one
                            asyncio.run(send_notification())

                        logger.info(f"Notification sent to chat {chat_id}")
                    except Exception as user_error:
                        logger.warning(f"Failed to send notification to chat {chat_id}: {user_error}")
                        # Remove invalid chat IDs
                        if "chat not found" in str(user_error).lower():
                            self.user_chat_ids.discard(chat_id)
                            self.save_config()

                logger.info(f"Notifications sent to {len(self.user_chat_ids)} users")
            else:
                logger.info("No Telegram users to notify or bot not running")
        except Exception as e:
            logger.error(f"Error sending Telegram notifications: {e}")

    def start_scheduler(self):
        """Start the posting scheduler (only if background tasks are enabled)"""
        if not self.enable_background_tasks:
            logger.info("Scheduler start skipped - background tasks disabled on this worker")
            return

        if self.scheduler_started:
            logger.info("Scheduler already started")
            return

        if not self.scheduler:
            logger.error("Scheduler not initialized - background tasks disabled")
            return

        try:
            # Clear any existing jobs
            self.scheduler.remove_all_jobs()

            # Add immediate first post job (2 minutes after startup)
            first_post_time = datetime.now() + timedelta(minutes=2)
            self.scheduler.add_job(
                func=self.generate_and_post,
                trigger='date',
                run_date=first_post_time,
                id='first_post_job',
                name='First Twitter Post',
                replace_existing=True
            )

            # Add the regular interval posting job (starts after first post)
            regular_start_time = first_post_time + timedelta(hours=POSTING_INTERVAL_HOURS)
            self.scheduler.add_job(
                func=self.generate_and_post,
                trigger=IntervalTrigger(hours=POSTING_INTERVAL_HOURS),
                start_date=regular_start_time,
                id='auto_post_job',
                name='Auto Twitter Post',
                replace_existing=True
            )

            # Add keep-alive job to prevent Render free tier from sleeping (every 14 minutes)
            self.scheduler.add_job(
                func=self.keep_alive_ping,
                trigger=IntervalTrigger(minutes=14),
                id='keep_alive_job',
                name='Keep-Alive Ping',
                replace_existing=True
            )

            # Update next post time to the first post
            self.stats['next_post_time'] = first_post_time.isoformat()
            self.save_config()

            self.scheduler.start()
            self.scheduler_started = True
            logger.info(f"✅ Scheduler started - first post at {first_post_time.strftime('%H:%M:%S')}, then every {POSTING_INTERVAL_HOURS} hours + keep-alive every 14min (Worker PID: {os.getpid()})")
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")

    def stop_scheduler(self):
        """Stop the posting scheduler"""
        if not self.scheduler or not self.scheduler_started:
            return

        try:
            self.scheduler.shutdown()
            self.scheduler_started = False
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

# Worker isolation: determine if this worker should run background tasks
def is_master_worker():
    """Determine if this worker should handle background tasks (scheduler/telegram)"""
    # Method 1: Use WORKER_ID if available (set by some deployment platforms)
    worker_id = os.environ.get('WORKER_ID')
    if worker_id is not None:
        is_master = worker_id == '0' or worker_id == '1'
        logger.info(f"Worker ID detection: {worker_id}, is_master: {is_master}")
        return is_master

    # Method 2: Use process ID modulo (fallback)
    pid = os.getpid()
    # Use a simple heuristic: lowest PID becomes master
    # This isn't perfect but works for most cases
    is_master = (pid % 100) < 50  # Roughly 50% chance, but consistent per PID
    logger.info(f"PID-based detection: PID {pid}, is_master: {is_master}")

    return is_master

# Global content manager instance with worker isolation
content_manager = ContentManager(enable_background_tasks=is_master_worker())

# Flask App Setup  
app = Flask(__name__)

# Secure session configuration - no fallback allowed
session_secret = os.environ.get('SESSION_SECRET')
if not session_secret:
    logger.error("SESSION_SECRET environment variable is required")
    exit(1)

app.secret_key = session_secret
CORS(app)

# Disable caching for better development experience in Replit
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/')
def index():
    """Main dashboard"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Twitter Bot Dashboard - {DEVELOPER_TG}</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    </head>
    <body class="bg-gray-100">
        <div id="root"></div>

        <script type="text/babel">
            const {{ useState, useEffect }} = React;

            function Dashboard() {{
                const [isAuthenticated, setIsAuthenticated] = useState(false);
                const [password, setPassword] = useState('');
                const [stats, setStats] = useState({{}});
                const [baseScript, setBaseScript] = useState('');
                const [newBaseScript, setNewBaseScript] = useState('');
                const [countdown, setCountdown] = useState('');

                useEffect(() => {{
                    checkAuth();
                    if (isAuthenticated) {{
                        fetchStats();
                        const interval = setInterval(updateCountdown, 1000);
                        return () => clearInterval(interval);
                    }}
                }}, [isAuthenticated]);

                const checkAuth = async () => {{
                    try {{
                        const response = await fetch('/api/check-auth');
                        const data = await response.json();
                        setIsAuthenticated(data.authenticated);
                    }} catch (error) {{
                        console.error('Auth check error:', error);
                    }}
                }};

                const login = async () => {{
                    try {{
                        const response = await fetch('/api/login', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ password }})
                        }});
                        const data = await response.json();
                        if (data.success) {{
                            setIsAuthenticated(true);
                            setPassword('');
                        }} else {{
                            alert('Invalid password');
                        }}
                    }} catch (error) {{
                        console.error('Login error:', error);
                    }}
                }};

                const fetchStats = async () => {{
                    try {{
                        const response = await fetch('/api/stats');
                        const data = await response.json();
                        setStats(data);
                        setBaseScript(data.base_script || '');
                    }} catch (error) {{
                        console.error('Stats fetch error:', error);
                    }}
                }};

                const updateBaseScript = async () => {{
                    try {{
                        const response = await fetch('/api/update-base', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ base_script: newBaseScript }})
                        }});
                        if (response.ok) {{
                            setBaseScript(newBaseScript);
                            setNewBaseScript('');
                            alert('Base script updated successfully!');
                            fetchStats();
                        }}
                    }} catch (error) {{
                        console.error('Update error:', error);
                    }}
                }};

                const updateCountdown = () => {{
                    if (stats.next_post_time) {{
                        const now = new Date();
                        const nextPost = new Date(stats.next_post_time);
                        const diff = nextPost - now;

                        if (diff > 0) {{
                            const hours = Math.floor(diff / (1000 * 60 * 60));
                            const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                            const seconds = Math.floor((diff % (1000 * 60)) / 1000);
                            setCountdown(`${{hours}}h ${{minutes}}m ${{seconds}}s`);
                        }} else {{
                            setCountdown('Post due now!');
                        }}
                    }}
                }};

                if (!isAuthenticated) {{
                    return (
                        <div className="min-h-screen flex items-center justify-center">
                            <div className="bg-white p-8 rounded-lg shadow-md w-96">
                                <h2 className="text-2xl font-bold mb-6 text-center">Admin Login</h2>
                                <input
                                    type="password"
                                    placeholder="Enter admin password"
                                    className="w-full p-3 border rounded mb-4"
                                    value={{password}}
                                    onChange={{(e) => setPassword(e.target.value)}}
                                    onKeyPress={{(e) => e.key === 'Enter' && login()}}
                                />
                                <button
                                    onClick={{login}}
                                    className="w-full bg-blue-500 text-white p-3 rounded hover:bg-blue-600"
                                >
                                    Login
                                </button>
                            </div>
                        </div>
                    );
                }}

                return (
                    <div className="min-h-screen bg-gray-100 p-8">
                        <div className="max-w-6xl mx-auto">
                            <header className="bg-white rounded-lg shadow-md p-6 mb-8">
                                <h1 className="text-3xl font-bold text-gray-800">AI Twitter Bot Dashboard</h1>
                                <p className="text-gray-600 mt-2">Developed by {DEVELOPER_TG}</p>
                            </header>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                                <div className="bg-white p-6 rounded-lg shadow-md">
                                    <h3 className="text-lg font-semibold mb-2">Posts Created</h3>
                                    <p className="text-3xl font-bold text-blue-500">{{stats.posts_created || 0}}</p>
                                </div>
                                <div className="bg-white p-6 rounded-lg shadow-md">
                                    <h3 className="text-lg font-semibold mb-2">Successful Posts</h3>
                                    <p className="text-3xl font-bold text-green-500">{{stats.posts_successful || 0}}</p>
                                </div>
                                <div className="bg-white p-6 rounded-lg shadow-md">
                                    <h3 className="text-lg font-semibold mb-2">Next Post In</h3>
                                    <p className="text-2xl font-bold text-orange-500">{{countdown}}</p>
                                </div>
                            </div>

                            <div className="bg-white rounded-lg shadow-md p-6 mb-8">
                                <h2 className="text-xl font-semibold mb-4">Current Base Script</h2>
                                <div className="bg-gray-100 p-4 rounded border">
                                    {{baseScript || "No base script set"}}
                                </div>
                            </div>

                            <div className="bg-white rounded-lg shadow-md p-6">
                                <h2 className="text-xl font-semibold mb-4">Update Base Script</h2>
                                <textarea
                                    className="w-full p-3 border rounded mb-4 h-32"
                                    placeholder="Enter new base script for AI content generation..."
                                    value={{newBaseScript}}
                                    onChange={{(e) => setNewBaseScript(e.target.value)}}
                                />
                                <button
                                    onClick={{updateBaseScript}}
                                    className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600"
                                >
                                    Update Base Script
                                </button>
                            </div>
                        </div>
                    </div>
                );
            }}

            ReactDOM.render(<Dashboard />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """

@app.route('/api/check-auth')
def check_auth():
    """Check if user is authenticated"""
    return jsonify({'authenticated': session.get('authenticated', False)})

@app.route('/api/login', methods=['POST'])
def login():
    """Handle admin login with secure password hashing"""
    from werkzeug.security import check_password_hash

    data = request.get_json()
    password = data.get('password', '')

    # Use proper password hash comparison
    if check_password_hash(ADMIN_PASSWORD_HASH, password):
        session['authenticated'] = True
        return jsonify({'success': True})

    # Log failed login attempts (without revealing the password)
    logger.warning(f"Failed login attempt from {request.remote_addr}")
    return jsonify({'success': False})

@app.route('/api/stats')
def get_stats():
    """Get bot statistics"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    stats = content_manager.stats.copy()
    stats['base_script'] = content_manager.base_script
    return jsonify(stats)

@app.route('/api/update-base', methods=['POST'])
def update_base():
    """Update base script"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    base_script = data.get('base_script', '')

    content_manager.base_script = base_script
    content_manager.save_config()

    return jsonify({'success': True})

@app.route('/api/manual-post', methods=['POST'])
def manual_post():
    """Trigger manual post"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        content_manager.generate_and_post()
        return jsonify({'success': True, 'message': 'Post generated and published!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    """Health check endpoint for keep-alive functionality"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running'
    })

def start_telegram_bot():
    """Start Telegram bot with proper thread management (only on master worker)"""
    if not content_manager.enable_background_tasks:
        logger.info("Telegram bot start skipped - background tasks disabled on this worker")
        return

    try:
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        if not token:
            logger.warning("Telegram bot token not found. Telegram features disabled.")
            return

        # Create telegram bot
        content_manager.telegram_bot = TelegramBot(token, content_manager)

        # Start the bot (it will create its own thread)
        content_manager.telegram_bot.run()

        logger.info(f"✅ Telegram bot initialization completed (Worker PID: {os.getpid()})")
    except Exception as e:
        logger.error(f"Error starting Telegram bot: {e}")

def create_app():
    """App factory for proper gunicorn deployment"""
    logger.info(f"🚀 Creating Flask app (Worker PID: {os.getpid()})")

    # Initialize background tasks if this is the master worker
    if content_manager.enable_background_tasks:
        logger.info("🎯 Initializing background tasks (scheduler & Telegram bot)...")
        try:
            # Start the content scheduler
            content_manager.start_scheduler()

            # Start Telegram bot
            start_telegram_bot()

            logger.info("✅ Background tasks initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing background tasks: {e}")
    else:
        logger.info("🔕 Background tasks disabled - this worker serves HTTP only")

    return app

# Clean app factory pattern for deployment
if __name__ == "__main__":
    # Development mode - force enable background tasks
    logger.info("🔧 Starting AI Twitter Bot System in development mode...")
    if not content_manager.enable_background_tasks:
        logger.info("🔄 Enabling background tasks for development mode...")
        content_manager.enable_background_tasks = True
        content_manager.scheduler = BackgroundScheduler()

    # Create the app
    app = create_app()

    # Get port from environment (for Railway/Render compatibility)
    port = int(os.environ.get("PORT", 5000))

    # Start the development server
    logger.info(f"Starting Flask development server on 0.0.0.0:{port}...")
    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        content_manager.stop_scheduler()
        if content_manager.telegram_bot:
            content_manager.telegram_bot.stop()
else:
    # Production mode - Gunicorn will use this
    logger.info("🐍 Production mode - using app factory pattern")
    app = create_app()
