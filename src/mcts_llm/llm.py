def chat_completion(messages, model):
    response = model.invoke(messages)
    return response
