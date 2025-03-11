import os
import discord
import logging

from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent

PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")

# A simple wrapper to simulate a discord.Message with a content attribute.
class FakeMessage:
    def __init__(self, content):
        self.content = content

@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints a message on the terminal when the bot successfully connects.
    """
    logger.info(f"{bot.user} has connected to Discord!")

@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.
    """
    if message.author == bot.user:
        return
        
    if message.author.bot:
        return
    
    if not message.content.startswith(PREFIX):
        return
        
    await bot.process_commands(message)

@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")

@bot.command(name="brainstorm", help="Get creative ideas from the Brainstormer agent.")
async def brainstorm(ctx, *, question=None):
    if question is None:
        await ctx.send("Please provide a question for the Brainstormer.")
        return
    
    conversation_log = [f"User: {question}"]
    brainstormer_prompt = build_brainstormer_context(conversation_log, 1, 1)
    response = await agent.run(FakeMessage(brainstormer_prompt))
    await ctx.send(f"**Brainstormer's Response:**\n{response}")

@bot.command(name="critique", help="Get feedback from the Critic agent.")
async def critique(ctx, *, idea=None):
    if idea is None:
        await ctx.send("Please provide an idea for the Critic to evaluate.")
        return
    
    conversation_log = [f"User: {idea}"]
    critic_prompt = build_critic_context(conversation_log, 1, 1)
    response = await agent.run(FakeMessage(critic_prompt))
    await ctx.send(f"**Critic's Response:**\n{response}")

@bot.command(name="help_roles", help="Shows available agent roles and their functions.")
async def help_roles(ctx):
    help_text = """
**Available Agent Roles:**

1. `!brainstorm <question>` - Get creative ideas and possibilities
   Example: `!brainstorm How can I improve my coding skills?`

2. `!critique <idea>` - Get evaluation and feedback on an idea
   Example: `!critique I want to learn programming by watching YouTube videos`

3. `!multiagent <question>` - Use all agents in a collaborative conversation
   Example: `!multiagent What's the best way to learn machine learning?`
"""
    await ctx.send(help_text)

def build_brainstormer_context(log, iteration, iteration_limit):
    context = "[System]\n"
    context += ("You are the Brainstormer agent. Your task is to generate creative and analytical ideas "
                "to address the user's query. Use the full conversation history below to build on previous ideas and refine your suggestions. "
                "Keep your response to 5 sentences or less, written in a single paragraph without bullet points or headings. "
                "Don't overcomplicate problems - when an answer is clear, answer directly without trying to find hidden meanings. "
                "Only use deeper reasoning when questions are genuinely complex or difficult. ")
    context += f"You have an iteration limit of {iteration_limit} rounds; if this is the final round, provide a summary of your best proposals.\n"
    context += "[Conversation History]\n"
    context += "\n".join(log) + "\n"
    context += "Brainstormer:\n"
    context += f"[Iteration Info]\nCurrent iteration: {iteration} of {iteration_limit}.\n"
    return context

def build_critic_context(log, iteration, iteration_limit):
    context = "[System]\n"
    context += ("You are the Critic agent. Your task is to evaluate the brainstormed ideas, flag any flaws or inaccuracies, "
                "and suggest improvements. Use the full conversation history below to ensure your feedback is thorough and relevant. "
                "Keep your response to 5 sentences or less, written in a single paragraph without bullet points or headings. "
                "Don't overcomplicate problems - when an answer is clear, focus on direct feedback without trying to find hidden meanings. "
                "Only use deeper critical analysis when questions are genuinely complex or difficult. ")
    context += f"The iteration limit is {iteration_limit} rounds; if this is the final round, emphasize the key points that need to be resolved.\n"
    context += "[Conversation History]\n"
    context += "\n".join(log) + "\n"
    context += "Critic:\n"
    context += f"[Iteration Info]\nCurrent iteration: {iteration} of {iteration_limit}.\n"
    return context

def build_synthesizer_context(log, iteration, iteration_limit):
    context = "[System]\n"
    context += ("You are the Synthesizer agent. Your task is to transform the brainstormed ideas and critiques into concrete, "
                "actionable steps or solutions. Focus on practicality and implementation details. "
                "Consider both the creative suggestions from the Brainstormer and the concerns raised by the Critic. "
                "Keep your response to 5 sentences or less, written in a single paragraph without bullet points or headings. "
                "Prioritize specific, implementable solutions over theoretical discussions. "
                "If technical details are relevant, include them concisely.")
    context += f"The iteration limit is {iteration_limit} rounds; if this is the final round, focus on the most viable solution.\n"
    context += "[Conversation History]\n"
    context += "\n".join(log) + "\n"
    context += "Synthesizer:\n"
    context += f"[Iteration Info]\nCurrent iteration: {iteration} of {iteration_limit}.\n"
    return context

def build_moderator_context(log, iteration, iteration_limit):
    context = "[System]\n"
    context += ("You are the Moderator agent. Your role is to manage the dialogue between the Brainstormer and Critic agents. "
                "Review the full conversation history below and determine who should contribute next. Ensure that the conversation stays "
                "on track and converges to a coherent answer. "
                "Keep your response to 5 sentences or less, written in a single paragraph without bullet points or headings. "
                "You don't need to use all iterations - if a clear answer has been reached, end the conversation early. It is critical that you do not overcomplicate or over-iterate.")
    context += f"The iteration limit is {iteration_limit} rounds.\n"
    context += "[Conversation History]\n"
    context += "\n".join(log) + "\n"
    context += "Moderator:\n"
    context += f"[Iteration Info]\nCurrent iteration: {iteration} of {iteration_limit}.\n"
    context += ("If the conversation should continue, provide your thoughts. "
                "If the conversation is complete, end with 'CONVO_OVER. SUMMARY: [your 2-3 sentence summary of the key points and conclusion. \
                    Note that this summary is to be displayed to the user, so you shouldn't mention things about the thought process, just the answer]'")
    return context

@bot.command(name="multiagent", help="Ask Mistral a question using a multi-agent conversation.")
async def multiagent(ctx, *, question=None):
    if question is None:
        await ctx.send("Please provide an input for the multiagent conversation.")
        return

    conversation_log = []
    conversation_log.append(f"User: {question}")
    iteration_limit = 3
    current_iteration = 1

    await ctx.send("**Starting multi-agent conversation...**")

    divider = "\n--------------------------------\n"
    
    while current_iteration <= iteration_limit:
        await ctx.send(f"**Iteration {current_iteration} of {iteration_limit}**")
        
        brainstormer_prompt = build_brainstormer_context(conversation_log, current_iteration, iteration_limit)
        brainstormer_response = await agent.run(FakeMessage(brainstormer_prompt))
        conversation_log.append("Brainstormer: " + brainstormer_response)
        await ctx.send("**Brainstormer:**\n" + brainstormer_response + divider)
        
        critic_prompt = build_critic_context(conversation_log, current_iteration, iteration_limit)
        critic_response = await agent.run(FakeMessage(critic_prompt))
        conversation_log.append("Critic: " + critic_response)
        await ctx.send("**Critic:**\n" + critic_response + divider)
        
        synthesizer_prompt = build_synthesizer_context(conversation_log, current_iteration, iteration_limit)
        synthesizer_response = await agent.run(FakeMessage(synthesizer_prompt))
        conversation_log.append("Synthesizer: " + synthesizer_response)
        await ctx.send("**Synthesizer:**\n" + synthesizer_response + divider)
        
        moderator_prompt = build_moderator_context(conversation_log, current_iteration, iteration_limit)
        moderator_response = await agent.run(FakeMessage(moderator_prompt))
        conversation_log.append("Moderator: " + moderator_response)
        await ctx.send("**Moderator:**\n" + moderator_response + divider)
        
        if "CONVO_OVER" in moderator_response:
            try:
                summary_text = moderator_response.split("SUMMARY:")[1].strip()
                await ctx.send("```\n== FINAL RESPONSE ==\n```\n**" + summary_text + "**\n```\nEnd of multi-agent conversation\n```")
            except IndexError:
                await ctx.send("```\n== FINAL RESPONSE ==\n```\n**Conversation complete. No detailed summary provided.**\n```\nEnd of multi-agent conversation\n```")
            break

        current_iteration += 1
    
    if current_iteration > iteration_limit:
        await ctx.send("**All iterations complete.**")

bot.run(token)
