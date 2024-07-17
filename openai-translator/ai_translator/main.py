import sys
import os
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, ConfigLoader, LOG
from model import GLMModel, OpenAIModel
from translator.pdf_translator import PDFTranslator
from translator.pdf_parser import PDFParser
from translator.writer import Writer
from book import Book, ContentType

def translate_pdf(pdf_file_path, file_format, target_language):
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()

    model_name = args.openai_model if args.openai_model else config['OpenAIModel']['model']
    api_key = args.openai_api_key if args.openai_api_key else config['OpenAIModel']['api_key']
    base_url = args.openai_api_key if args.openai_api_key else config['OpenAIModel']['base_url']
    model = OpenAIModel(model=model_name, api_key=api_key, base_url=base_url)

    translator = PDFTranslator(model)
    translator.translate_pdf(pdf_file_path, file_format, target_language)
    
    markdown_content = ''
    for page in translator.book.pages:
        for content in page.contents:
            if content.status:
                if content.content_type == ContentType.TEXT:
                    markdown_content += content.translation + '\n\n'
                elif content.content_type == ContentType.TABLE:
                    table = content.translation
                    header = '| ' + ' | '.join(str(column) for column in table.columns) + ' |' + '\n'
                    separator = '| ' + ' | '.join(['---'] * len(table.columns)) + ' |' + '\n'
                    body = '\n'.join(['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in table.values.tolist()]) + '\n\n'
                    markdown_content += header + separator + body
        markdown_content += '---\n\n'

    return markdown_content
             

def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# PDF Translator")
        
        with gr.Column():
            pdf_file_path = gr.Textbox(label="PDF File Path", placeholder="Enter the path to the PDF file")
            file_format = gr.Dropdown(label="File Format", choices=["PDF", "Markdown"], value="PDF")
            target_language = gr.Textbox(label="Target Language", placeholder="Enter the target language (e.g., 日语)")
            
            translate_button = gr.Button("Translate PDF")
            result_textbox = gr.Textbox(label="Translated PDF", interactive=False, lines=20)
            
        translated_pdf = translate_button.click(translate_pdf, inputs=[pdf_file_path, file_format, target_language], outputs=[result_textbox])

    demo.launch()

if __name__ == "__main__":
    gradio_app()