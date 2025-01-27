import requests
from config import SERPER_API_KEY

def web_search(query, num_results=5):
    """
    Perform a web search using Serper API.

    Args:
        query (str): Search query string.
        num_results (int): Number of search results to retrieve.

    Returns:
        list: List of search results with title, link, and snippet.
    """
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY is not set. Please provide your API key.")

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    payload = {"q": query}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad HTTP status codes

        results = response.json().get("organic", [])
        if not results:
            return [{"title": "No Results Found", "link": "", "snippet": ""}]

        return [
            {
                "title": result.get("title", "No Title"),
                "link": result.get("url", "No URL"),
                "snippet": result.get("snippet", "No Description"),
            }
            for result in results[:num_results]
        ]

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"An error occurred while making the API request: {e}")