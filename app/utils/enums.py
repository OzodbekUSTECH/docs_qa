from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    
    
class DocumentType(str, Enum):
    FINANCIAL = "FINANCIAL"
    INVOICE = "INVOICE"
    CONTRACT = "CONTRACT"
    COO = "COO" # ceritificate of origin
    COA = "COA" # certificate of analysis
    COW = "COW" # certificate of weight
    COQ = "COQ" # certificate of quality
    BL = "BL" # bill of lading
    LC = "LC" # letter of credit
    OTHER = "OTHER"