from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, DataTable, Label, Log, Input
from textual.reactive import reactive
from textual import events
import numpy as np
from typing import List, Dict, Tuple

class InputArea(Static):
    """Editable input area with token display"""
    
    def __init__(self):
        super().__init__()
        self.tokens = []
        
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter text here...", id="text-input")
        yield ScrollableContainer(
            Log(),
            DataTable()
        )
    
    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns("Position", "Token", "ID", "Is Locked")
        
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle text input changes"""
        # Here we would tokenize the input
        # For now, just split on spaces
        text = event.value
        mock_tokens = [(word, hash(word) % 10000) for word in text.split()]
        self.update_tokens(mock_tokens)
    
    def update_tokens(self, new_tokens, locked_positions=None):
        """Update the token display with new tokens"""
        if locked_positions is None:
            locked_positions = set()
            
        log = self.query_one(Log)
        table = self.query_one(DataTable)
        
        # Update the readable text log
        log.clear()
        log.write(" ".join(token for token, _ in new_tokens))
        
        # Clear rows while keeping columns
        table.clear()
        table.add_columns("Position", "Token", "ID", "Is Locked")
        
        # Add the new rows
        for pos, (token, token_id) in enumerate(new_tokens):
            is_locked = "🔒" if pos in locked_positions else ""
            table.add_row(str(pos), token, str(token_id), is_locked)
        
        self.tokens = new_tokens

class TokenView(Static):
    """A view showing different token filtering perspectives"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.tokens: List[Tuple[str, float, np.ndarray]] = []  # token, prob, embedding
        self.concept_vectors: Dict[str, np.ndarray] = {}
        
    def compute_vector_distances(self, token_embedding: np.ndarray) -> Dict[str, float]:
        """Compute cosine distances to each concept vector"""
        distances = {}
        for concept_name, concept_vec in self.concept_vectors.items():
            # Normalize vectors
            token_norm = token_embedding / np.linalg.norm(token_embedding)
            concept_norm = concept_vec / np.linalg.norm(concept_vec)
            # Compute cosine similarity
            distance = 1 - np.dot(token_norm, concept_norm)
            distances[concept_name] = distance
        return distances

class TopKView(TokenView):
    """Shows top k most likely tokens"""
    
    def compose(self) -> ComposeResult:
        yield DataTable()
    
    def update_tokens(self, tokens: List[Tuple[str, float, np.ndarray]], k: int = 5):
        """Update with top k tokens by probability"""
        sorted_tokens = sorted(tokens, key=lambda x: x[1], reverse=True)[:k]
        table = self.query_one(DataTable)
        table.clear()
        table.add_columns("Token", "Probability")
        for token, prob, _ in sorted_tokens:
            table.add_row(token, f"{prob:.4f}")

class BottomKView(TokenView):
    """Shows bottom k least likely tokens"""
    
    def compose(self) -> ComposeResult:
        yield DataTable()
    
    def update_tokens(self, tokens: List[Tuple[str, float, np.ndarray]], k: int = 5):
        """Update with bottom k tokens by probability"""
        sorted_tokens = sorted(tokens, key=lambda x: x[1])[:k]
        table = self.query_one(DataTable)
        table.clear()
        table.add_columns("Token", "Probability")
        for token, prob, _ in sorted_tokens:
            table.add_row(token, f"{prob:.4f}")

class ConceptDistanceView(TokenView):
    """Shows tokens closest/furthest from concept vectors"""
    
    def compose(self) -> ComposeResult:
        yield DataTable()
    
    def update_tokens(self, tokens: List[Tuple[str, float, np.ndarray]]):
        """Update showing distances to concept vectors"""
        table = self.query_one(DataTable)
        table.clear()
        
        # Create columns for token and each concept
        columns = ["Token"] + list(self.concept_vectors.keys())
        table.add_columns(*columns)
        
        # Compute and display distances
        for token, prob, embedding in tokens:
            distances = self.compute_vector_distances(embedding)
            row = [token] + [f"{distances[c]:.4f}" for c in self.concept_vectors.keys()]
            table.add_row(*row)

class TokenExplorer(App):
    """Interactive token exploration interface"""
    
    CSS = """
    TokenView {
        height: 1fr;
        border: solid green;
    }
    
    InputArea {
        height: 2fr;
        border: solid blue;
    }
    
    Input {
        dock: top;
        margin: 1;
    }
    
    Log {
        height: 1fr;
        border: solid yellow;
        padding: 1;
        background: $surface;
        color: $text;
    }
    
    #input-container {
        height: 2fr;
    }
    
    #top-container {
        height: 3fr;
        layout: horizontal;
    }
    
    #bottom-container {
        height: 2fr;
    }
    """

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Vertical(id="input-container"):
            yield InputArea()
        with Horizontal(id="top-container"):
            yield TopKView("Top Tokens")
            yield BottomKView("Bottom Tokens")
        with Horizontal(id="bottom-container"):
            yield ConceptDistanceView("Concept Distances")
        yield Footer()

    def on_mount(self) -> None:
        """Set up initial state when app starts."""
        # Mock input tokens for initial view
        input_area = self.query_one(InputArea)
        input_area.update_tokens([
            ("The", 464),
            ("quick", 4789),
            ("brown", 7891),
            ("fox", 2345)
        ], locked_positions={1})  # Lock "quick" for demo
        
        # Mock next token predictions
        token_data = [
            ("jumps", 0.25, np.random.rand(768)),
            ("runs", 0.15, np.random.rand(768)),
            ("leaps", 0.10, np.random.rand(768)),
            ("walks", 0.08, np.random.rand(768)),
            ("sits", 0.05, np.random.rand(768)),
        ]
        
        # Mock concept vectors
        concept_vectors = {
            "technical": np.random.rand(768),
            "emotional": np.random.rand(768),
            "formal": np.random.rand(768)
        }
        
        # Update all token prediction views
        for view in self.query(TokenView):
            view.concept_vectors = concept_vectors
            view.update_tokens(token_data)

if __name__ == "__main__":
    app = TokenExplorer()
    app.run()