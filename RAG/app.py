import os
import gradio as gr
import glob
import json
from LLM_usage import JokeGeneratorBot
from prompt import PROMPT

# Define the base path
base_path = "./Data" # "/mnt/d/Semester7/NLP/RAG/Data"

# Global variables for reuse
bot = None
docs = []

# Function to initialize the bot (called once during deployment)
def initialize_bot(api_key):
    global bot, docs

    # Set the API key
    os.environ['GROQ_API_KEY'] = api_key

    # Load documents (done once)
    if not docs:  # Only load if docs are not already loaded
        for path in glob.glob(f"{base_path}/*_processed.json"):
            with open(path, 'r') as f:
                docs.extend(json.load(f))

    # Initialize the bot
    bot = JokeGeneratorBot(docs, PROMPT)

# Function to handle joke generation
def generate_joke_interface(user_message, regime, bm_koef):
    global bot

    if bm_koef < 0 or bm_koef > 1:
        return "Error: BM25 coefficient should be in the range [0, 1]. Please provide a valid value.", [["", ""]] 

    # Check if the bot is initialized
    if bot is None:
        return "Error: Bot is not initialized. Please provide an API key during deployment.", []

    # Map the selected regime to the corresponding flags
    bm25_only = regime == "BM25 Only"
    semantic_only = regime == "Semantic Only"
    scores_combination = regime == "Scores Combination"

    # Call the bot to generate a joke
    result = bot.generate_joke(
        user_message, n=40, bm25_only=bm25_only, semantic_only=semantic_only, scores_combination=scores_combination, bm_koef=bm_koef
    )

    # Format the context as a DataFrame for better display
    context = []
    for doc in result["Context"]:
        context.extend([[k, v] for k, v in doc.items()])
        context.append(["", ""])  # Add empty space between documents

    return result["Response"], context

# Create a setup interface for API key input
def setup_interface(api_key):
    initialize_bot(api_key)
    return "Joke Generator initialized successfully!"

# Create Gradio interface
setup_demo = gr.Interface(
    fn=setup_interface,
    inputs=[gr.Textbox(label="Enter your GROQ API Key")],
    outputs=[gr.Textbox(label="Setup Status")],
    title="Setup Joke Generator",
    description="Initialize the Joke Generator Bot by providing the GROQ API key. (If there is a connection error just submit the key again. It will work.)",
)

regime_options = ["BM25 Only", "Semantic Only", "Scores Combination"]
joke_demo = gr.Interface(
    fn=generate_joke_interface,
    inputs=[
        gr.Textbox(label="Enter your message"),
        gr.Radio(choices=regime_options, label="Choose Regime", value="Scores Combination"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.75, label="BM25 Coefficient"),
    ],
    outputs=[
        gr.Textbox(label="Generated Joke"),
        gr.Dataframe(headers=["Key", "Value"], label="Context"),
    ],
    title="Joke Generator",
    description="Generate jokes based on your input message(Only in English :( )). Select a retrieval regime and view the context used.\
                Be careful, the jokes can be offensive! Try to write a message that is related to the joke you want to hear.\
                (tell me a joke and its title about... or tell me a one liner about...). Sometimes bot works bad :(\
                In this case, try to rewrite a message and send again. Or close the window and enter\
                the link again, after reinitialize joke generator with API KEY.\
                Or try to change the regime or BM25 Coefficient.\
                BM25 Coefficient is used to balance the BM25 and semantic scores(It is active only in Scores Combination mode). Semantic scores are multiplied by (1 - BM25 Coefficient).\
                If you want to use only BM25 or semantic scores, select the corresponding regime or set it to 0.0 or 1.0. respectively.",
)

# In order to divide a big string line into a couple of lines, I can place a backslash at the end of the line 

# Combine setup and main interfaces into a tabbed app
demo = gr.TabbedInterface(
    [setup_demo, joke_demo],
    ["Setup", "Joke Generator"]
)

# Launch the interface
# demo.launch()
demo.launch(share=True)
