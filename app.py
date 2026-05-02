import gradio as gr

from chain import load_chain, ask

chain = load_chain()


def respond(message, history):

    if not message.strip():
        return "Enter a question"

    result = ask(chain, message)

    output = result["answer"]

    if result["sources"]:
        output += "\n\nSources: " + ", ".join(result["sources"])

    return output


demo = gr.ChatInterface(
    fn=respond,
    title="RAG Chatbot",
    description="Ask questions from your documents"
)

demo.launch()