from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import cv2
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from models import GameState, Hand
from card_detector import CardDetector
from strategy import BasicStrategy
from card_counter import CardCounter


class GameManager:
    """Manages the game state and coordinates all components"""

    def __init__(self):
        self.detector = CardDetector()
        self.strategy = BasicStrategy()
        self.counter = CardCounter(num_decks=6)
        self.current_game_state: Optional[GameState] = None
        self.hand_history: List[Dict] = []
        self.is_camera_active = False

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single camera frame"""
        # Detect cards
        detected_cards = self.detector.detect_cards(frame)

        if detected_cards:
            # Update count with new cards
            new_cards = [d.card for d in detected_cards]
            self.counter.update_count(new_cards)

            # Create game state
            # For now, assume first 2 cards are player's, 3rd is dealer's
            if len(detected_cards) >= 3:
                player_cards = [detected_cards[0].card, detected_cards[1].card]
                dealer_card = detected_cards[2].card

                player_hand = Hand(cards=player_cards)

                # Get basic strategy decision
                decision = self.strategy.get_decision(
                    player_hand, dealer_card, can_double=len(player_cards) == 2
                )

                # Check for count-based deviations
                deviations = self.counter.get_strategy_deviations(
                    player_hand.total, dealer_card.value, player_hand.is_soft
                )

                self.current_game_state = GameState(
                    player_hand=player_hand,
                    dealer_card=dealer_card,
                    recommended_action=decision,
                    true_count=self.counter.get_true_count(),
                    running_count=self.counter.running_count,
                )

                return {
                    "status": "success",
                    "game_state": self.current_game_state.dict(),
                    "count_stats": self.counter.get_count_stats(),
                    "deviations": deviations,
                    "detected_cards": len(detected_cards),
                }

        return {
            "status": "no_cards_detected",
            "count_stats": self.counter.get_count_stats(),
        }

    def save_hand(self, result: str, bet: float = 0):
        """Save completed hand to history"""
        if self.current_game_state:
            self.hand_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "player_cards": [
                        c.dict() for c in self.current_game_state.player_hand.cards
                    ],
                    "dealer_card": self.current_game_state.dealer_card.dict(),
                    "player_total": self.current_game_state.player_hand.total,
                    "recommended_action": self.current_game_state.recommended_action,
                    "result": result,  # "win", "loss", "push", "blackjack"
                    "bet": bet,
                    "true_count": round(self.current_game_state.true_count, 2),
                }
            )

    def reset_shoe(self):
        """Reset counter for new shoe"""
        self.counter.reset()

    def get_statistics(self) -> Dict:
        """Get session statistics"""
        if not self.hand_history:
            return {"total_hands": 0}

        wins = sum(1 for h in self.hand_history if h["result"] == "win")
        losses = sum(1 for h in self.hand_history if h["result"] == "loss")
        pushes = sum(1 for h in self.hand_history if h["result"] == "push")

        total_bet = sum(h["bet"] for h in self.hand_history)
        total_won = sum(
            h["bet"] * 1.5 if h["result"] == "blackjack" else h["bet"]
            for h in self.hand_history
            if h["result"] in ["win", "blackjack"]
        )
        total_lost = sum(h["bet"] for h in self.hand_history if h["result"] == "loss")

        return {
            "total_hands": len(self.hand_history),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / len(self.hand_history) * 100, 1)
            if self.hand_history
            else 0,
            "total_bet": total_bet,
            "net_profit": total_won - total_lost,
            "current_count": self.counter.get_count_stats(),
        }


# Create FastAPI app
app = FastAPI(title="Blackjack CV Helper API")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create game manager instance
game_manager = GameManager()


@app.get("/")
async def root():
    return JSONResponse(
        content={"message": "Blackjack CV Helper API", "version": "0.1.0"},
        status_code=200,
    )


@app.get("/stats")
async def get_statistics():
    """Get current session statistics"""
    return JSONResponse(content=game_manager.get_statistics(), status_code=200)


@app.post("/reset-shoe")
async def reset_shoe():
    """Reset card counter for new shoe"""
    game_manager.reset_shoe()
    return JSONResponse(
        content={
            "message": "Shoe reset",
            "count_stats": game_manager.counter.get_count_stats(),
        },
        status_code=200,
    )


@app.post("/save-hand")
async def save_hand(result: str, bet: float = 0):
    """Save completed hand result"""
    game_manager.save_hand(result, bet)
    return JSONResponse(
        content={
            "message": "Hand saved",
            "total_hands": len(game_manager.hand_history),
        },
        status_code=200,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time camera feed processing"""
    await websocket.accept()

    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)  # Use default camera

        if not cap.isOpened():
            await websocket.send_json({"error": "Camera not found"})
            return

        game_manager.is_camera_active = True

        while True:
            # Check for messages from client
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(message)

                if data.get("command") == "stop":
                    break
                elif data.get("command") == "reset_shoe":
                    game_manager.reset_shoe()

            except asyncio.TimeoutError:
                pass

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                continue

            # Process frame
            result = game_manager.process_frame(frame)

            # Send results to frontend
            await websocket.send_json(result)

            # Small delay to control frame rate
            await asyncio.sleep(0.1)  # 10 FPS

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        game_manager.is_camera_active = False
        cap.release()


@app.get("/camera-status")
async def camera_status():
    """Check if camera is active"""
    return JSONResponse(
        content={"is_active": game_manager.is_camera_active}, status_code=200
    )


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "timestamp": datetime.now().isoformat()},
        status_code=200,
    )
