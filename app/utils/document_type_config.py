"""Конфигурация важных параграфов для разных типов документов"""
from app.utils.enums import DocumentType
from typing import Dict, List

# Маппинг типа документа на список важных параграфов/секций
DOCUMENT_TYPE_IMPORTANT_CLAUSES: Dict[DocumentType, List[str]] = {
    DocumentType.CONTRACT: [
        "parties", "buyer", "seller", "price", "payment terms", "delivery terms",
        "quality specifications", "quantity", "warranty", "liability", "termination",
        "force majeure", "dispute resolution", "governing law", "penalties"
    ],
    DocumentType.INVOICE: [
        "invoice number", "date", "buyer", "seller", "items", "quantities",
        "unit prices", "total amount", "tax", "payment terms", "due date",
        "bank details", "currency"
    ],
    DocumentType.BL: [  # Bill of Lading
        "shipper", "consignee", "notify party", "vessel", "voyage number",
        "port of loading", "port of discharge", "description of goods",
        "quantity", "weight", "freight", "date", "signature"
    ],
    DocumentType.LC: [  # Letter of Credit
        "issuing bank", "beneficiary", "applicant", "amount", "currency",
        "expiry date", "documents required", "shipment terms", "payment terms",
        "partial shipments", "transshipment"
    ],
    DocumentType.COA: [  # Certificate of Analysis
        "product name", "batch number", "test results", "specifications",
        "test methods", "date of analysis", "laboratory", "signature"
    ],
    DocumentType.COO: [  # Certificate of Origin
        "exporter", "importer", "country of origin", "goods description",
        "hs code", "date", "signature", "certifying authority"
    ],
    DocumentType.COW: [  # Certificate of Weight
        "shipper", "consignee", "gross weight", "net weight", "tare weight",
        "number of packages", "date", "weighing location", "signature"
    ],
    DocumentType.COQ: [  # Certificate of Quality
        "product description", "quality specifications", "test results",
        "inspection date", "inspector", "certification body", "signature"
    ],
    DocumentType.FINANCIAL: [
        "balance sheet", "income statement", "cash flow", "assets", "liabilities",
        "equity", "revenue", "expenses", "profit", "loss", "audit opinion"
    ],
    DocumentType.OTHER: [
        "key terms", "important information", "dates", "parties", "amounts",
        "conditions", "requirements"
    ],
}


def get_important_clauses_for_type(document_type: DocumentType) -> List[str]:
    """Возвращает список важных параграфов для типа документа"""
    return DOCUMENT_TYPE_IMPORTANT_CLAUSES.get(document_type, DOCUMENT_TYPE_IMPORTANT_CLAUSES[DocumentType.OTHER])

