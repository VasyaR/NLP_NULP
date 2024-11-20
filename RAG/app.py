import gradio as gr

from LLM_usage import JokeGeneratorBot

def generate_joke(user_message):
    bot = JokeGeneratorBot()
    return bot.generate_joke(user_message)

demo = gr.Interface(fn=generate_joke, inputs="text", outputs="text")
demo.launch()
# demo.launch(share=True)