class Retriever:

    def __init__(self, docs: [str]) -> None:
        self.docs = []

    def get_docs(self, user_message: str, n: int = 3) -> [str]:
        return self.docs