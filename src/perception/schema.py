from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

@dataclass
class UIElement:
    id: str  # Unique identifier (could be semantic or index-based)
    type: str  # button, input, text, icon, etc.
    bbox: BoundingBox
    text: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict) # e.g., state: disabled, checked
    confidence: float = 1.0
    source: str = "vlm"  # vlm, ocr, layout, merged

@dataclass
class UISceneGraph:
    """
    Represents the structural truth of the screen.
    Independent of the Planner's intent.
    """
    elements: List[UIElement] = field(default_factory=list)
    resolution: Tuple[int, int] = (0, 0)
    
    def add_element(self, element: UIElement):
        self.elements.append(element)
    
    def get_element_by_id(self, uid: str) -> Optional[UIElement]:
        return next((e for e in self.elements if e.id == uid), None)
