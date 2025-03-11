import os
from mistralai import Mistral
import discord
import asyncio

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = "You are a helpful assistant."


class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

        self.client = Mistral(api_key=MISTRAL_API_KEY)

    async def run(self, message: discord.Message):
        # The simplest form of an agent
        # Send the message's content to Mistral's API and return Mistral's response

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message.content},
        ]

        try:
            response = await self.client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=messages,
            )
            return response.choices[0].message.content

        except Exception as e:
            if "rate limit exceeded" in str(e).lower():
                await asyncio.sleep(5)
                return await self.run(message)

            return "Error: Unable to process request due to API rate limit."