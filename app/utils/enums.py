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
    
    
class ExtractionFieldType(str, Enum):
    CLAUSE = "CLAUSE"      # clause-like text
    TEXT = "TEXT"          # generic text field
    TABLE = "TABLE"        # table extraction
    NUMBER = "NUMBER"
    DATE = "DATE"
    BOOL = "BOOL"
    # extend later without versioning
    
class FieldOccurrence(str, Enum):
    OPTIONAL_ONCE = "OPTIONAL_ONCE"
    OPTIONAL_MULTIPLE = "OPTIONAL_MULTIPLE"
    REQUIRED_ONCE = "REQUIRED_ONCE"
    REQUIRED_MULTIPLE = "REQUIRED_MULTIPLE"


class ExtractionBy(str, Enum):
    DOCUMENT_AI = "DOCUMENT_AI"
    GEMINI_AI = "GEMINI_AI"