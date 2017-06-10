import datetime

from report.markdown_document_builder import MarkdownDocumentBuilder
from report.default_report import create_default_report


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")