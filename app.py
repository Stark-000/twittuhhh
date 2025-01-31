import os
import random
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, constr
from dotenv import load_dotenv
import aiohttp
import tweepy
from contextlib import asynccontextmanager
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

USE_MOCK = os.getenv("USE_MOCK", "0").lower() in ["1", "true"]
TWITTER_PROXY = os.getenv("TWITTER_PROXY")

def init_twitter_client():
    logger.info("Initializing Twitter client...")
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    
    try:
        client = tweepy.Client(
            bearer_token=bearer_token,
            wait_on_rate_limit=True,
            timeout=10
        )
        
        logger.debug("Testing Twitter authentication...")
        me = client.get_me(user_fields=['username'], user_auth=False)
        
        if not me or not me.data:
            raise Exception("Failed to get user data")
            
        logger.info(f"Twitter authentication successful. Connected as: {me.data.username}")
        return client
            
    except tweepy.TooManyRequests as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        return None
    except tweepy.Unauthorized as e:
        logger.error(f"Authentication failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Twitter client initialization failed: {str(e)}")
        return None

twitter_client = init_twitter_client()
if twitter_client is None:
    logger.warning("Failed to initialize Twitter client, falling back to mock mode")
    USE_MOCK = True

logger.info(f"Application running in {'mock' if USE_MOCK else 'real'} mode")

class UserCreate(BaseModel):
    user_id: str

class AccountAdd(BaseModel):
    handle: str
    alias: Optional[str] = None

class DailySummary(BaseModel):
    account: str
    alias: Optional[str]
    summary: str
    sentiment: str
    category: str
    tweet_count: int
    time_span: str

class HealthCheck(BaseModel):
    status: str
    twitter_api: bool
    groq_api: bool
    mock_mode: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    connector = aiohttp.TCPConnector(ssl=False)
    app.http_session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30)
    )

    app.users_db = {}

    yield

    await app.http_session.close()

app = FastAPI(
    title="Twitter Monitor Pro",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs"
)

async def analyze_with_grok(prompt: str) -> str:
    if USE_MOCK:
        return random.choice([
            "Positive outlook on technology advancements",
            "Neutral discussion about current events",
            "Exciting sports updates"
        ])
    
    try:
        async with app.http_session.post(
            "https://api.groq.com/v1/chat/completions",
            json={
                "model": "mixtral-8x7b-32768",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            proxy=TWITTER_PROXY
        ) as response:
            if response.status != 200:
                error_data = await response.json()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Groq API error: {error_data.get('error', 'Unknown error')}"
                )
            data = await response.json()
            return data['choices'][0]['message']['content'].strip()
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

async def get_real_tweets(handle: str) -> List[dict]:
    if USE_MOCK:
        return [
            {
                "text": f"Mock tweet for @{handle} about technology #tech",
                "created_at": datetime.now() - timedelta(hours=2),
                "likes": random.randint(100, 500),
                "retweets": random.randint(10, 50),
                "replies": random.randint(1, 20),
                "hashtags": ["tech", "innovation"],
                "context": ["Technology", "Business"]
            },
            {
                "text": f"Another mock tweet from @{handle} about AI #ai",
                "created_at": datetime.now() - timedelta(hours=5),
                "likes": random.randint(50, 200),
                "retweets": random.randint(5, 25),
                "replies": random.randint(1, 10),
                "hashtags": ["ai", "future"],
                "context": ["Technology", "Artificial Intelligence"]
            }
        ]
    
    try:
        logger.debug(f"Fetching user details for @{handle}")
        user = twitter_client.get_user(
            username=handle,
            user_fields=['id', 'name', 'username'],
            user_auth=False
        )
        
        if not user or not user.data:
            logger.error(f"User @{handle} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Twitter user @{handle} not found"
            )

        logger.debug(f"Fetching tweets for user ID: {user.data.id}")
        tweets = twitter_client.get_users_tweets(
            user.data.id,
            max_results=20,
            tweet_fields=[
                "created_at",
                "public_metrics",
                "context_annotations",
                "entities"
            ],
            exclude=["retweets", "replies"],
            user_auth=False
        )

        if not tweets or not tweets.data:
            logger.warning(f"No tweets found for @{handle}")
            return []

        return [{
            "text": tweet.text,
            "created_at": tweet.created_at,
            "likes": tweet.public_metrics["like_count"],
            "retweets": tweet.public_metrics["retweet_count"],
            "replies": tweet.public_metrics["reply_count"],
            "hashtags": [tag["tag"] for tag in tweet.entities.get("hashtags", [])] if hasattr(tweet, "entities") else [],
            "context": [
                anno["domain"]["name"]
                for anno in (getattr(tweet, "context_annotations", []) or [])
            ]
        } for tweet in tweets.data]

    except tweepy.TooManyRequests as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        raise HTTPException(
            status_code=429,
            detail=f"Twitter API rate limit reached. Reset at: {e.response.headers.get('x-rate-limit-reset')}"
        )
    except tweepy.Unauthorized as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Twitter API authentication failed. Please check credentials."
        )
    except Exception as e:
        logger.error(f"Twitter API error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Twitter API Error: {str(e)}"
        )

async def get_tweets(handle: str) -> List[dict]:
    return await get_real_tweets(handle)



@app.post("/users/", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    if user.user_id in app.users_db:
        raise HTTPException(status_code=409, detail="User exists")
    
    app.users_db[user.user_id] = {
        "monitored_accounts": {},
        "metadata": user.dict()
    }
    return {"message": "User created"}

@app.post("/users/{user_id}/accounts/")
async def add_account(user_id: str, account: AccountAdd):
    if user_id not in app.users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    app.users_db[user_id]["monitored_accounts"][account.handle] = {
        "alias": account.alias,
        "added_at": datetime.now()
    }
    return {"message": f"Added @{account.handle}"}

@app.get("/users/{user_id}/summary/", response_model=List[DailySummary])
async def get_summary(user_id: str):
    if user_id not in app.users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    summaries = []
    for handle, account in app.users_db[user_id]["monitored_accounts"].items():
        tweets = await get_tweets(handle)
        
        tweet_context = "\n".join([
            f"Tweet: {t['text']}\n"
            f"Engagement: {t['likes']} likes, {t['retweets']} retweets\n"
            f"Topics: {', '.join(t['hashtags'] + t['context'])}\n"
            for t in tweets[:10]
        ])
        
        summary, sentiment, category = await asyncio.gather(
            analyze_with_grok(
                f"Analyze and summarize these tweets from @{handle}. "
                f"Focus on main themes and key messages: {tweet_context}"
            ),
            analyze_with_grok(
                f"Analyze the sentiment of these tweets. "
                f"Consider engagement metrics and language used: {tweet_context}"
            ),
            analyze_with_grok(
                f"Categorize the main focus areas of these tweets "
                f"(e.g., Tech, Business, Personal): {tweet_context}"
            )
        )
        
        summaries.append(DailySummary(
            account=handle,
            alias=account["alias"],
            summary=summary,
            sentiment=sentiment,
            category=category,
            tweet_count=len(tweets),
            time_span="Last 24h"
        ))
    
    return summaries

@app.get("/health/", response_model=HealthCheck)
async def health_check():
    twitter_status = False
    groq_status = False
    
    try:
        if not USE_MOCK:
            twitter_client.get_me()
            twitter_status = True
            
            async with app.http_session.get(
                "https://api.groq.com/v1/health",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                proxy=TWITTER_PROXY
            ) as response:
                groq_status = response.status == 200
    except:
        pass

    return HealthCheck(
        status="OK",
        twitter_api=twitter_status,
        groq_api=groq_status,
        mock_mode=USE_MOCK
    )

@app.delete("/users/{user_id}/accounts/{handle}/")
async def remove_account(user_id: str, handle: str):
    if user_id not in app.users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if handle not in app.users_db[user_id]["monitored_accounts"]:
        raise HTTPException(status_code=404, detail="Account not monitored")
    
    del app.users_db[user_id]["monitored_accounts"][handle]
    return {"message": f"Removed @{handle}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
