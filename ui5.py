from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, TextArea
from textual import events
import numpy as np
from typing import List, Dict, Tuple

class TokenAnalysisView(Static):
    """Shows token analysis in various forms"""
    
    def compose(self) -> ComposeResult:
        """Create tables for token analysis"""
        yield DataTable(id="token-table")
        with Horizontal():
            yield DataTable(id="top-tokens")
            yield DataTable(id="concept-distances")
    
    def on_mount(self) -> None:
        """Initialize the data tables"""
        # Current tokens table
        token_table = self.query_one("#token-table", DataTable)
        token_table.add_columns("Position", "Token", "ID", "Is Locked")
        
        # Top tokens table
        top_tokens = self.query_one("#top-tokens", DataTable)
        top_tokens.add_columns("Token", "Probability")
        
        # Concept distances table
        distances = self.query_one("#concept-distances", DataTable)
        distances.add_columns("Token", "Technical", "Emotional", "Formal")

    def update_current_tokens(self, tokens: List[Tuple[str, int]], locked_positions: set = None):
        """Update the current tokens display"""
        if locked_positions is None:
            locked_positions = set()
            
        table = self.query_one("#token-table", DataTable)
        table.clear()
        
        for pos, (token, token_id) in enumerate(tokens):
            is_locked = "🔒" if pos in locked_positions else ""
            table.add_row(str(pos), token, str(token_id), is_locked)

    def update_predictions(self, tokens: List[Tuple[str, float, np.ndarray]]):
        """Update the token predictions"""
        top_tokens = self.query_one("#top-tokens", DataTable)
        top_tokens.clear()
        
        # Show top 5 predictions
        sorted_tokens = sorted(tokens, key=lambda x: x[1], reverse=True)[:5]
        for token, prob, _ in sorted_tokens:
            top_tokens.add_row(token, f"{prob:.4f}")

class TokenExplorer(App):
    """Interactive token exploration interface"""
    
    CSS = """
    TextArea {
        height: 1fr;
        border: solid green;
        margin: 1;
    }
    
    TokenAnalysisView {
        height: 2fr;
        border: solid blue;
    }
    
    #token-table {
        height: 1fr;
        margin: 1;
    }
    
    #top-tokens, #concept-distances {
        width: 1fr;
        height: 1fr;
        margin: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield TextArea()
        yield TokenAnalysisView()
        yield Footer()

    def on_mount(self) -> None:
        """Set up initial state when app starts."""
        # Set up mock data
        analysis = self.query_one(TokenAnalysisView)
        analysis.update_current_tokens([
            ("The", 464),
            ("quick", 4789),
            ("brown", 7891),
            ("fox", 2345)
        ])
        
        analysis.update_predictions([
            ("jumps", 0.25, np.random.rand(768)),
            ("runs", 0.15, np.random.rand(768)),
            ("leaps", 0.10, np.random.rand(768)),
            ("walks", 0.08, np.random.rand(768)),
            ("sits", 0.05, np.random.rand(768)),
        ])

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes"""
        # Mock tokenization - in practice, use your MLX tokenizer
        tokens = [(word, hash(word) % 10000) 
                 for word in event.text.split()]
        
        analysis = self.query_one(TokenAnalysisView)
        analysis.update_current_tokens(tokens)

if __name__ == "__main__":
    app = TokenExplorer()
    app.run()
