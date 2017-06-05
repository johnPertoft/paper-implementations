
class MarkdownDocumentBuilder:
    def __init__(self):
        self.file_content = ""

    def add_header(self, title, level=1):
        assert 1 <= level <= 6
        self.file_content += "{} {}\n".format("#" * level, title)

    def add_table(self, rows):
        max_cols = max(len(r) for r in rows)
        assert max_cols >= 2

        # Table header
        self.file_content += "| " * max_cols + "|\n"
        self.file_content += "| - " * max_cols + "|\n"

        # Table content
        table_rows = ["| {} |".format(" | ".join(str(e) for e in row)) for row in rows]
        self.file_content += "\n".join(table_rows) + "\n"

    def add_images(self, image_paths):
        # TODO: Add some options for formatting this.
        markdown_images = ["![alt text]({})".format(path) for path in image_paths]
        self.file_content += "\n".join(markdown_images) + "\n"

    def build(self, path):
        with open(path, "w") as f:
            f.write(self.file_content)
