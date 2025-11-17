from dishka import Provider, Scope, provide, provide_all

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PictureDescriptionVlmOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from openai import AsyncOpenAI
from app.core.config import settings

from google.genai import Client
from google.genai.client import AsyncClient
 
class UtilsProvider(Provider):
   
    scope = Scope.APP
    
    utils = provide_all(
    )
    
    
    @provide
    def provide_google_genai_client(self) -> AsyncClient:
        return Client(api_key=settings.GOOGLE_API_KEY).aio
    
    @provide
    def provide_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)  
   
    @provide
    def provide_document_converter(self) -> DocumentConverter:
        pipeline_options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
            ),
            do_ocr=True,
            images_scale=2,
            
            ocr_batch_size=4,
            layout_batch_size=64,
            table_batch_size=4,
            
            # pdf 
            # do_picture_description=True,
            # picture_description_options=PictureDescriptionVlmOptions(
            #     repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
            #     prompt="Describe this picture in 3-5 sentences.",
            # ),
            # generate_picture_images=True
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
