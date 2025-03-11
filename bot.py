import os
import discord
import logging
from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent
from googlesearch import search
from collections import defaultdict

PREFIX = "!"

logger = logging.getLogger("discord")

load_dotenv()

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

agent = MistralAgent()

token = os.getenv("DISCORD_TOKEN")

# Session memory to store conversation history per user
user_memories = defaultdict(list)
MAX_MEMORY_LENGTH = 10  # Maximum number of interactions to remember per user

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

def get_user_memory(user_id):
    """Retrieve the conversation memory for a specific user"""
    return user_memories[user_id]

def add_to_memory(user_id, role, content):
    """Add a new message to the user's conversation memory"""
    user_memories[user_id].append(f"{role}: {content}")
    # Keep memory within size limit
    if len(user_memories[user_id]) > MAX_MEMORY_LENGTH:
        user_memories[user_id].pop(0)

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
    
    user_id = ctx.author.id
    add_to_memory(user_id, "User", question)
    
    conversation_log = get_user_memory(user_id)
    brainstormer_prompt = build_brainstormer_context(conversation_log, 1, 1)
    response = await agent.run(FakeMessage(brainstormer_prompt))
    
    add_to_memory(user_id, "Brainstormer", response)
    await ctx.send(f"**Brainstormer's Response:**\n{response}")

@bot.command(name="critique", help="Get feedback from the Critic agent.")
async def critique(ctx, *, idea=None):
    if idea is None:
        await ctx.send("Please provide an idea for the Critic to evaluate.")
        return
    
    user_id = ctx.author.id
    add_to_memory(user_id, "User", idea)
    
    conversation_log = get_user_memory(user_id)
    critic_prompt = build_critic_context(conversation_log, 1, 1)
    response = await agent.run(FakeMessage(critic_prompt))
    
    add_to_memory(user_id, "Critic", response)
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

def build_search_context(log, iteration, iteration_limit):
    """
    First prompt: The Search Agent decides if it needs to do a Google search.
    It may respond with 'DO_SEARCH: <query>' if it wants you to gather info,
    or it may respond with a direct answer if no search is needed.
    """
    context = "[System]\n"
    context += (
        "You are the Search agent. Your role is to determine if a web search is required "
        "to gather factual information. If you decide to do a search, respond with a line "
        "starting with 'DO_SEARCH:' followed by the query terms. If no search is needed, "
        "just provide a direct answer.\n"
    )
    context += f"Iteration limit: {iteration_limit}.\n"
    context += "[Conversation History]\n"
    context += "\n".join(log) + "\n"
    context += "Search Agent:\n"
    context += f"[Iteration Info]\nCurrent iteration: {iteration} of {iteration_limit}.\n"
    return context


def build_search_results_context(log, search_results, iteration, iteration_limit):
    """
    Second prompt: The Search Agent processes and organizes search results
    to produce a comprehensive, insight-driven summary.
    """
    context = "[System]\n"
    context += (
        "You are the Search agent. Below is organized information returned by your Google search. "
        "Synthesize a comprehensive answer that directly addresses the user's original question. "
        "Focus on extracting key insights and presenting them in a logical flow. "
        "Provide specific details and facts from multiple sources when available. "
        "Your response should be 3-5 sentences, coherent and complete. "
    )
    
    # Extract the original user question from the conversation log
    user_question = ""
    for entry in reversed(log):
        if entry.startswith("User:"):
            user_question = entry[6:].strip()
            break
    
    context += f"\n[Original Question]\n{user_question}\n\n"
    context += f"[Iteration Info]\nCurrent iteration: {iteration} of {iteration_limit}.\n\n"
    context += "[Conversation History]\n"
    context += "\n".join(log) + "\n\n"
    
    # Process and organize search results more effectively
    context += "[Processed Search Results]\n"
    
    # Group results by potential relevance
    highly_relevant = []
    somewhat_relevant = []
    
    # Basic relevance sorting based on keyword matching with the question
    question_keywords = set(user_question.lower().split())
    for res in search_results[:5]:
        title_and_desc = (res.title + " " + res.description).lower()
        keyword_matches = sum(1 for keyword in question_keywords if keyword in title_and_desc)
        
        if keyword_matches >= 2:  # More matching keywords = higher relevance
            highly_relevant.append(res)
        else:
            somewhat_relevant.append(res)
    
    # Present the most relevant results first
    context += "Most Relevant Information:\n"
    for i, res in enumerate(highly_relevant, 1):
        context += f"{i}. {res.title}\n   Key Points: {res.description}\n\n"
    
    context += "Additional Information:\n"
    for i, res in enumerate(somewhat_relevant, 1):
        context += f"{i}. {res.title}\n   Details: {res.description}\n\n"
    
    # Extract key facts if possible
    context += "Key Facts:\n"
    all_text = " ".join(res.description for res in search_results[:5])
    # Extract sentences that contain question keywords
    sentences = [s.strip() for s in all_text.split('.') if s.strip()]
    fact_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in question_keywords):
            fact_sentences.append(sentence)
    
    # Add the extracted facts (up to 5)
    for i, fact in enumerate(fact_sentences[:5], 1):
        context += f"- {fact}.\n"
    
    context += "\nSearch Agent:\n"
    return context

@bot.command(name="searchagent", help="Use the search agent to gather info from Google.")
async def searchagent_cmd(ctx, *, question=None):
    """
    Demonstration command that triggers the Search Agent:
      1) Agent decides whether to search.
      2) We run the search if requested.
      3) Agent summarizes the search results.
    """
    if question is None:
        await ctx.send("Please provide a query or question for the Search Agent.")
        return

    user_id = ctx.author.id
    add_to_memory(user_id, "User", question)
    
    conversation_log = get_user_memory(user_id)
    iteration_limit = 2
    current_iteration = 1

    search_prompt = build_search_context(conversation_log, current_iteration, iteration_limit)
    initial_response = await agent.run(FakeMessage(search_prompt))
    
    add_to_memory(user_id, "SearchAgent", initial_response)

    if "DO_SEARCH:" in initial_response:
        search_query = initial_response.split("DO_SEARCH:")[1].strip()

        await ctx.send(f"**Search Agent**: Performing Google search for: `{search_query}`")

        raw_results = list(search(search_query, num_results=5, advanced=True))

        current_iteration += 1
        search_results_prompt = build_search_results_context(
            conversation_log, raw_results, current_iteration, iteration_limit
        )
        final_summary = await agent.run(FakeMessage(search_results_prompt))
        
        add_to_memory(user_id, "SearchAgent", final_summary)
        await ctx.send("**Search Agent Summary**:\n" + final_summary)
    else:
        await ctx.send("**Search Agent** did not request a search. Response:\n" + initial_response)


def build_brainstormer_context(log, iteration, iteration_limit):
    context = "[System]\n"
    context += ("You are the Brainstormer agent. Your task is to generate creative and analytical ideas "
                "to address the user's query. Use the full conversation history below to build on previous ideas and refine your suggestions. "
                "If search results are included in the conversation, treat them as accurate and current information. "
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
                "If search results are included in the conversation, treat them as accurate and current information - do not question their validity. "
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
                "If search results are included in the conversation, incorporate this factual information into your synthesis. "
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
    context += ("You are the Moderator agent. Your role is to manage the dialogue between the Brainstormer and Critic and Synthesizer agents. "
                "Review the full conversation history below and determine who should contribute next. Ensure that the conversation stays "
                "on track and converges to a coherent answer. "
                "If search results are included in the conversation, ensure they are properly incorporated as factual information. "
                "Keep your response to 5 sentences or less, written in a single paragraph without bullet points or headings. "
                "You don't need to use all iterations - if a clear answer has been reached, end the conversation early. It is critical that you do not overcomplicate or over-iterate, \
                    but also do not end the conversation prematurely if what the Synthesizer provided is not yet complete.")
    context += f"The iteration limit is {iteration_limit} rounds.\n"
    context += "[Conversation History]\n"
    context += "\n".join(log) + "\n"
    context += "Moderator:\n"
    context += f"[Iteration Info]\nCurrent iteration: {iteration} of {iteration_limit}.\n"
    context += ("If the conversation should continue, provide your thoughts. "
                "If the conversation is complete, end with 'CONVO_OVER. SUMMARY: [your summary of the key points and conclusion. \
                    If the question was simple, keep this to 1-2 sentences. If the question was complex/technical, you can write 5-6 sentences. \
                    Note that this summary is to be displayed to the user, so you shouldn't mention things about the thought process, just the answer]'")
    return context

@bot.command(name="multiagent", help="Ask Mistral a question using a multi-agent conversation. Use --search to include web search results.")
async def multiagent(ctx, *, question=None):
    if question is None:
        await ctx.send("Please provide an input for the multiagent conversation.")
        return
        
    use_search = False
    if question.startswith("--search "):
        use_search = True
        question = question[len("--search "):].strip()
    
    user_id = ctx.author.id
    add_to_memory(user_id, "User", question)
    
    conversation_log = get_user_memory(user_id)
    iteration_limit = 3
    current_iteration = 1

    await ctx.send("**Starting multi-agent conversation...**")
    
    if use_search:
        await ctx.send("**Searching for information first...**")
        search_prompt = build_search_context(conversation_log, 1, 2)
        search_decision = await agent.run(FakeMessage(search_prompt))
        
        if "DO_SEARCH:" in search_decision:
            search_query = search_decision.split("DO_SEARCH:")[1].strip()
            await ctx.send(f"**Search Agent**: Performing Google search for: `{search_query}`")
            
            raw_results = list(search(search_query, num_results=5, advanced=True))
            
            search_results_prompt = build_search_results_context(
                conversation_log, raw_results, 2, 2
            )
            search_summary = await agent.run(FakeMessage(search_results_prompt))
            
            search_date = "as of today's date"
            search_preamble = f"[The following information was gathered from a Google search {search_date} and should be considered accurate factual information]"
            add_to_memory(user_id, "SearchResults", f"{search_preamble}\n{search_summary}")
            await ctx.send(f"**Search Results**:\n{search_summary}")

    divider = "\n--------------------------------\n"
    
    while current_iteration <= iteration_limit:
        await ctx.send(f"**Iteration {current_iteration} of {iteration_limit}**")
        
        conversation_log = get_user_memory(user_id)
        
        brainstormer_prompt = build_brainstormer_context(conversation_log, current_iteration, iteration_limit)
        brainstormer_response = await agent.run(FakeMessage(brainstormer_prompt))
        add_to_memory(user_id, "Brainstormer", brainstormer_response)
        await ctx.send("**Brainstormer:**\n" + brainstormer_response + divider)
        
        critic_prompt = build_critic_context(conversation_log, current_iteration, iteration_limit)
        critic_response = await agent.run(FakeMessage(critic_prompt))
        add_to_memory(user_id, "Critic", critic_response)
        await ctx.send("**Critic:**\n" + critic_response + divider)
        
        synthesizer_prompt = build_synthesizer_context(conversation_log, current_iteration, iteration_limit)
        synthesizer_response = await agent.run(FakeMessage(synthesizer_prompt))
        add_to_memory(user_id, "Synthesizer", synthesizer_response)
        await ctx.send("**Synthesizer:**\n" + synthesizer_response + divider)
        
        moderator_prompt = build_moderator_context(conversation_log, current_iteration, iteration_limit)
        moderator_response = await agent.run(FakeMessage(moderator_prompt))
        add_to_memory(user_id, "Moderator", moderator_response)
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

@bot.command(name="clear_memory", help="Clear your conversation history with the bot.")
async def clear_memory(ctx):
    user_id = ctx.author.id
    if user_id in user_memories:
        user_memories[user_id] = []
        await ctx.send("Your conversation history has been cleared.")
    else:
        await ctx.send("You don't have any conversation history to clear.")

bot.run(token)