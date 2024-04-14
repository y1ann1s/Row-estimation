import openai
import config

# Set your OpenAI API key in a secure manner, e.g., environment variables or a config file
openai.api_key = config.openai_api_key

def ask_chatgpt(question, messages):
    """Ask ChatGPT a question and append the interaction to the chat history, acting as a rowing coach."""
    # Define the role of ChatGPT as a rowing coach
# Check if the messages list is empty
    if len(messages) == 0:
        role_context = "You are an experienced rowing coach. You provide expert advice on rowing techniques, training programs, and competition preparation. If anyone asks who you are or similar questions, your response should be that you are the virtual brain of George Gioupis."
        # Append the role context as a system message, not a user message
        messages.append({"role": "system", "content": role_context})
    else:
     # Append the user's question to messages
     messages.append({"role": "user", "content": question})


    try:
        response = openai.chat.completions.create (
          model="gpt-3.5-turbo",
          messages=messages,
            max_tokens=150,
            temperature=0.7,
            #stop=["\nHuman:", "\nAI:"],
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.6
            )
        answer = response.choices[0].message.content
        messages.append({"role": "system", "content": answer})

        #return f"Gioupis Says: {answer}"
        return answer, messages
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return "Sorry, the coaches brain is busy thinking other things.", messages

