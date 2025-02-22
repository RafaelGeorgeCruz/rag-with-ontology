from langchain_google_genai import ChatGoogleGenerativeAI
from chat_base import ChatPDFBase


class ChatPDF(ChatPDFBase):
    def __init__(self, model_name="gemini-2.0-flash"):
        super().__init__()
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
        )

    def generate_response(self, prompt):
        return self.model.predict(prompt)
