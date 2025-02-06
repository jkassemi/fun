"""
A UI for exploring geometric transformation concepts.
Visualize and manipulate embedding spaces and attention patterns.
"""

import mlx.core as mx
from mlx_lm import load, generate
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, TextArea, ProgressBar
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
            "Position", "prev(n=1)|p", "Token", "ID", "Is Locked",
            "top(n=1)|v", "top(n=1)|p",
            "top(n=2)|v", "top(n=2)|p", 
            "top(n=3)|v", "top(n=3)|p",
            "top(n=4)|v", "top(n=4)|p",
            "top(n=5)|v", "top(n=5)|p"
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
            
            # Calculate prev token probability (mock for now)
            prev_prob = "0.00" if pos == 0 else "0.85"
            table.add_row(str(pos), prev_prob, token, str(token_id), is_locked, *pred_values)

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
            "Position", "prev(n=1)|p", "Token", "ID", "Is Locked",
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
            
        table = self.query_one("#bottom-token-table", DataTable)
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
            
            # Calculate prev token probability (mock for now)
            prev_prob = "0.00" if pos == 0 else "0.85"
            table.add_row(str(pos), prev_prob, token, str(token_id), is_locked, *pred_values)


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

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._inactivity_timer = None
        self._progress_timer = None

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield TextArea()
        yield ProgressBar(total=100, show_percentage=True)
        yield TopTokenAnalysisView()
        yield BottomTokenAnalysisView()
        yield Footer()

    def on_mount(self) -> None:
        """Set up initial state when app starts."""
        # Set up analysis views
        top_analysis = self.query_one(TopTokenAnalysisView)
        bottom_analysis = self.query_one(BottomTokenAnalysisView)
        text_area = self.query_one(TextArea)
        
        # Set initial text
        text_area.text = "The quick"
        
        # Sample tokens - just the ones from our text
        tokens = [
            ("The", 464),
            ("quick", 4789)
        ]
        
        # Mock top predictions for just our two tokens
        top_predictions = [
            [("is", 0.95), ("was", 0.85), ("and", 0.80), ("has", 0.78), ("will", 0.75)],
            [("brown", 0.90), ("red", 0.85), ("lazy", 0.80), ("small", 0.78), ("big", 0.75)]
        ]
        
        # Mock bottom predictions for just our two tokens
        bottom_predictions = [
            [("xyz", 0.05), ("123", 0.04), ("@#$", 0.03), ("...", 0.02), ("???", 0.01)],
            [("9876", 0.06), ("qwer", 0.05), ("asdf", 0.04), ("zxcv", 0.03), ("jklm", 0.02)]
        ]
        
        top_analysis.update_current_tokens(tokens, predictions=top_predictions)
        bottom_analysis.update_current_tokens(tokens, predictions=bottom_predictions)

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes"""
        # Cancel existing timers if any
        if self._inactivity_timer:
            self._inactivity_timer.cancel()
        if self._progress_timer:
            self._progress_timer.cancel()
            
        # Reset progress bar
        progress = self.query_one(ProgressBar)
        progress.update(progress=0)
            
        # Create new timers
        self._inactivity_timer = asyncio.create_task(self._handle_inactivity())
        self._progress_timer = asyncio.create_task(self._update_progress())
        
        # Get the TextArea widget and its current content
        text_area = self.query_one(TextArea)
        current_text = text_area.text
        
    async def _update_progress(self) -> None:
        """Update progress bar during inactivity period"""
        progress = self.query_one(ProgressBar)
        for i in range(30):  # 30 steps over 3 seconds
            await asyncio.sleep(0.1)  # 100ms per step
            progress.update(progress=((i+1) * 100) // 30)
            
    async def _handle_inactivity(self) -> None:
        """Called when text area has been inactive for 3 seconds"""
        await asyncio.sleep(3.0)
        progress = self.query_one(ProgressBar)
        progress.update(progress=100)
        print("Text area inactive for 3 seconds")
        
        # Get current text
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
    checkpoint: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model, tokenizer = load(path_or_hf_repo=checkpoint)
    app = TokenExplorer(model, tokenizer)
    # # Load the model
    # self.model, self.tokenizer = load(path_or_hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
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
