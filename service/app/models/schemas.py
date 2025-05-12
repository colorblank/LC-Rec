from pydantic import BaseModel


# Renamed from InputData for clarity, serves requirement 3
class BatchVectorInput(BaseModel):
    data: list[list[float]]  # Assuming a list of vectors (list of floats)


# New Pydantic model for single text item (requirement 2)
class SingleTextItem(BaseModel):
    text: str


# New Pydantic model for batch text items (requirement 1)
class BatchTextItems(BaseModel):
    texts: list[str]
