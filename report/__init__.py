import datetime

from report.markdown_document_builder import MarkdownDocumentBuilder


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")