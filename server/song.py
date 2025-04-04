from mcp.server.fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def top_song(genre: str) -> str:
    """Get the most popular song for a given genre.
    Supported genres: pop, rock, jazz, classical, hiphop, kpop.

    Args:
        genre: song genre
    """
    top_songs = {
        "pop": "Blinding Lights - The Weeknd",
        "rock": "Bohemian Rhapsody - Queen",
        "jazz": "So What - Miles Davis",
        "classical": "Canon in D - Pachelbel",
        "hiphop": "SICKO MODE - Travis Scott",
        "kpop": "Dynamite - BTS"
    }
    return top_songs.get(genre, f"No top song found for genre: {genre}")

if __name__ == "__main__":
    mcp.run(transport='stdio')