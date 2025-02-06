"""
MOCK UI FOR EXPLORATION ONLY

This is a prototype UI for exploring geometric transformation concepts.
Currently uses mock data instead of real model integration.
The goal is to experiment with different ways of visualizing and 
manipulating embedding spaces and attention patterns.
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, TextArea
from textual import events
import numpy as np
from typing import List, Dict, Tuple

class TopTokenAnalysisView(Static):
    """Shows top token predictions"""
    
    def compose(self) -> ComposeResult:
        """Create tables for token analysis"""
        yield Static("Top Predictions", classes="table-header")
        yield DataTable(id="top-token-table")
    
    def on_mount(self) -> None:
        """Initialize the data tables"""
        token_table = self.query_one("#top-token-table", DataTable)
        token_table.add_columns(
            "Position", "Token", "ID", "Is Locked",
            "top(n=1)|v", "top(n=1)|p",
            "top(n=2)|v", "top(n=2)|p", 
            "top(n=3)|v", "top(n=3)|p",
            "top(n=4)|v", "top(n=4)|p",
            "top(n=5)|v", "top(n=5)|p"
        )

class BottomTokenAnalysisView(Static):
    """Shows bottom token predictions"""
    
    def compose(self) -> ComposeResult:
        """Create tables for token analysis"""
        yield Static("Bottom Predictions", classes="table-header")
        yield DataTable(id="bottom-token-table")
    
    def on_mount(self) -> None:
        """Initialize the data tables"""
        token_table = self.query_one("#bottom-token-table", DataTable)
        token_table.add_columns(
            "Position", "Token", "ID", "Is Locked",
            "bott(n=1)|v", "bott(n=1)|p",
            "bott(n=2)|v", "bott(n=2)|p", 
            "bott(n=3)|v", "bott(n=3)|p",
            "bott(n=4)|v", "bott(n=4)|p",
            "bott(n=5)|v", "bott(n=5)|p"
        )

    def update_current_tokens(self, tokens: List[Tuple[str, int]], locked_positions: set = None, predictions: List[List[Tuple[str, float]]] = None):
        """Update the current tokens display"""
        if locked_positions is None:
            locked_positions = set()
        if predictions is None:
            predictions = [[("", 0.0)] * 5] * len(tokens)  # Default empty predictions
            
        table = self.query_one("#top-token-table", DataTable)
        table.clear()
        
        for pos, ((token, token_id), token_predictions) in enumerate(zip(tokens, predictions)):
            is_locked = "🔒" if pos in locked_positions else ""
            
            # Pad predictions to ensure 5 entries
            while len(token_predictions) < 5:
                token_predictions.append(("", 0.0))
            
            # Flatten predictions into alternating token/probability values
            pred_values = []
            for pred_token, pred_prob in token_predictions[:5]:
                pred_values.extend([pred_token, f"{pred_prob:.4f}"])
            
            table.add_row(str(pos), token, str(token_id), is_locked, *pred_values)


class TokenExplorer(App):
    """Interactive token exploration interface"""
    
    CSS = """
    .table-header {
        background: $accent;
        color: $text;
        text-align: center;
    }
    
    TextArea {
        height: 1fr;
        border: solid green;
        margin: 1;
    }
    
    TopTokenAnalysisView, BottomTokenAnalysisView {
        height: 1fr;
        border: solid blue;
    }
    
    #top-token-table, #bottom-token-table {
        height: 1fr;
        margin: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield TextArea()
        yield TopTokenAnalysisView()
        yield BottomTokenAnalysisView()
        yield Footer()

    def on_mount(self) -> None:
        """Set up initial state when app starts."""
        # Set up mock data
        top_analysis = self.query_one(TopTokenAnalysisView)
        bottom_analysis = self.query_one(BottomTokenAnalysisView)
        
        # Sample tokens
        tokens = [
            ("The", 464),
            ("quick", 4789),
            ("brown", 7891),
            ("fox", 2345)
        ]
        
        # Mock top predictions
        top_predictions = [
            [("is", 0.95), ("was", 0.85), ("and", 0.80), ("has", 0.78), ("will", 0.75)],
            [("brown", 0.90), ("red", 0.85), ("lazy", 0.80), ("small", 0.78), ("big", 0.75)],
            [("fox", 0.90), ("dog", 0.80), ("cat", 0.75), ("bear", 0.70), ("wolf", 0.65)],
            [("jumps", 0.85), ("runs", 0.75), ("leaps", 0.70), ("walks", 0.68), ("sits", 0.65)]
        ]
        
        # Mock bottom predictions
        bottom_predictions = [
            [("xyz", 0.05), ("123", 0.04), ("@#$", 0.03), ("...", 0.02), ("???", 0.01)],
            [("9876", 0.06), ("qwer", 0.05), ("asdf", 0.04), ("zxcv", 0.03), ("jklm", 0.02)],
            [("!!!!", 0.04), ("****", 0.03), ("____", 0.02), ("^^^^", 0.01), (">>>>", 0.005)],
            [("0000", 0.03), ("1111", 0.02), ("2222", 0.01), ("3333", 0.005), ("4444", 0.001)]
        ]
        
        top_analysis.update_current_tokens(tokens, predictions=top_predictions)
        bottom_analysis.update_current_tokens(tokens, predictions=bottom_predictions)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes"""
        # Get the TextArea widget and its current content
        text_area = self.query_one(TextArea)
        current_text = text_area.text
        
        # Mock tokenization - in practice, use your MLX tokenizer
        tokens = [(word, hash(word) % 10000) 
                 for word in current_text.split()]
        
        # Mock predictions for each token - in practice, get these from your MLX model
        example_predictions = [
            ("jumps", 0.25), ("runs", 0.15), ("leaps", 0.10), 
            ("walks", 0.08), ("sits", 0.05)
        ]
        mock_predictions = [example_predictions for _ in range(len(tokens))]
        
        top_analysis = self.query_one(TopTokenAnalysisView)
        bottom_analysis = self.query_one(BottomTokenAnalysisView)
        
        top_analysis.update_current_tokens(tokens, predictions=mock_predictions)
        bottom_analysis.update_current_tokens(tokens, predictions=mock_predictions)

if __name__ == "__main__":
    import sys
    app = TokenExplorer()
    
    # Allow --test flag for automated testing
    if "--test" in sys.argv:
        print("Running in test mode")
        
        # Create test app and debug widget tree
        test_app = TokenExplorer()
        print("\nApp created")
        
        # Try composing
        widgets = test_app.compose()
        print("\nCompose returned:", list(widgets))
        
        # Debug widget tree
        print("\nWidget tree before mount:")
        print(test_app.tree)
        
        # Try mounting
        test_app.post_message(events.Mount())
        print("\nWidget tree after mount:")
        print(test_app.tree)
        
        # Debug query
        try:
            text_area = test_app.query_one(TextArea)
            print("\nFound TextArea:", text_area)
        except Exception as e:
            print(f"\nError finding TextArea: {type(e).__name__}: {str(e)}")
    else:
        app.run()


# ideas when training:
#
# 1. you could probably find xml tags to define how to perceive the associated content useful. 
#           def <1>my_function</1>:
#               pass
#
#    think of this like an extension of the `ai?`, 'ai', 'ai!' marking concept. leads to more
#    effective context discovery.
#
# 2. 
