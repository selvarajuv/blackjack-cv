import uvicorn
import sys
from pathlib import Path

# Add src to path so imports work
sys.path.append(str(Path(__file__).parent))


def main():
    """Run the Blackjack CV Helper server"""
    print("🎰 Starting Blackjack CV Helper API...")
    print("📡 Server will be available at http://localhost:8000")
    print("📊 API docs available at http://localhost:8000/docs")
    print("🎥 Make sure your camera is connected!")
    print("\nPress CTRL+C to stop the server\n")

    # Run the server
    uvicorn.run(
        "api:app",
        host="127.0.0.1",  # localhost only for development
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )


if __name__ == "__main__":
    main()
