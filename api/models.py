from typing import Any

from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any]