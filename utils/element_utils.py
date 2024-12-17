from typing import List, Any
from unstructured.documents.elements import Text, Title, Table

def load_elements(element_strings: List[str]) -> List[Any]:
    """Reconstruct elements from stored strings."""
    elements = []
    for e_str in element_strings:
        if "Title:" in e_str:
            elements.append(Title(text=e_str.replace("Title: ", "")))
        elif "Table:" in e_str:
            elements.append(Table(text=e_str.replace("Table: ", "")))
        else:
            elements.append(Text(text=e_str))
    return elements
