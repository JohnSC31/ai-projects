class Document:
    def __init__(self, text, filename, author, date, week):
        self.text = text
        self.filename = filename
        self.author = author
        self.date = date
        self.week = week
    def get_text(self):
        return self.text
    def get_metadata(self):
        return f"Documento: {self.filename}, hecho por: {self.author}, de la clase del: {self.date}, semana #{self.week} del periodo lectivo."
