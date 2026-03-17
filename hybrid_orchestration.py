"""
Hybrid Orchestration: Agentkube-Mini + LangGraph Sub-Agents

This module demonstrates how to use Agentkube-Mini for top-level orchestration
while keeping LangGraph sub-agents for specialized work (music catalog, invoices).

Key Insight:
- LangGraph excels at complex agent reasoning with tool-calling (ReAct).
- Agentkube-Mini excels at simple, deterministic orchestration (DAG).
- Combining them gives you simplicity at the top and power where needed.
"""

import uuid
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Agentkube-Mini imports
from agentkube_mini import Agent, TaskGraph, Runtime, EventBus

# For this example, we simulate LangGraph sub-agents.
# In real usage, you'd import the compiled graphs from multi_agent.ipynb.


# ============================================================================
# SIMULATED IN-MEMORY STORES (would be real databases/caches in production)
# ============================================================================

class MemoryStore:
    """Simple in-memory storage for user preferences."""
    def __init__(self):
        self.data = {}
    
    def get(self, key: tuple, subkey: str) -> Optional[Dict]:
        """Retrieve data."""
        if key in self.data and subkey in self.data[key]:
            return self.data[key][subkey]
        return None
    
    def put(self, key: tuple, subkey: str, value: Dict) -> None:
        """Store data."""
        if key not in self.data:
            self.data[key] = {}
        self.data[key][subkey] = value


# Shared memory store for user preferences (long-term memory)
memory_store = MemoryStore()

# Sample database of customers (simulate Chinook DB)
CUSTOMER_DATABASE = {
    "1": {"name": "John Doe", "email": "john@example.com", "phone": "+55 (12) 3923-5555"},
    "2": {"name": "Jane Smith", "email": "jane@example.com", "phone": "+1 (204) 452-6452"},
    "123": {"name": "Test User", "email": "test@example.com", "phone": "+1 (555) 123-4567"},
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_customer_id(text: str) -> Optional[str]:
    """Extract customer ID, email, or phone from text."""
    # Try direct customer ID (all digits)
    match = re.search(r'\b(\d+)\b', text)
    if match:
        customer_id = match.group(1)
        if customer_id in CUSTOMER_DATABASE:
            return customer_id
    
    # Try email
    match = re.search(r'[\w.-]+@[\w.-]+\.\w+', text)
    if match:
        email = match.group(0)
        for cid, data in CUSTOMER_DATABASE.items():
            if data["email"] == email:
                return cid
    
    # Try phone
    match = re.search(r'[\+\d\s\(\)\-]+', text)
    if match:
        phone = match.group(0)
        for cid, data in CUSTOMER_DATABASE.items():
            if data["phone"] == phone:
                return cid
    
    return None


def format_preferences(prefs_data: Optional[Dict]) -> str:
    """Format stored preferences into a readable string."""
    if not prefs_data:
        return "No saved preferences."
    music_prefs = prefs_data.get("music_preferences", [])
    if music_prefs:
        return f"Music Preferences: {', '.join(music_prefs)}"
    return "No saved preferences."


def extract_music_preferences(question: str, response: str) -> list:
    """Extract music-related keywords from question and response."""
    combined = (question + " " + response).lower()
    # Simple keyword extraction (in real system, use LLM for this)
    artists = []
    keywords = ["u2", "rolling stones", "pink floyd", "coldplay", "coldplay", "queen", "beatles"]
    for keyword in keywords:
        if keyword in combined:
            artists.append(keyword.title())
    return artists


# ============================================================================
# AGENTKUBE-MINI AGENT FUNCTIONS (Orchestration Layer)
# ============================================================================

def verify_customer_id(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    AGENT 1: Extract and validate customer ID from input.
    
    Input: {"question": "My ID is 123. What albums by U2?"}
    Output: {"customer_id": "123", "question": "...", "verified": True}
    """
    question = input_data.get("question", "")
    customer_id = extract_customer_id(question)
    
    return {
        "customer_id": customer_id,
        "question": question,
        "verified": bool(customer_id),
    }


def load_user_memory(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    AGENT 2: Load user preferences from long-term memory.
    Depends on: verify_customer_id (needs customer_id).
    
    Input: {"customer_id": "123"}
    Output: {"preferences": "Music Preferences: U2, Rolling Stones"}
    """
    customer_id = input_data.get("customer_id")
    preferences = ""
    
    if customer_id:
        namespace = ("memory_profile", customer_id)
        stored = memory_store.get(namespace, "user_memory")
        preferences = format_preferences(stored)
    
    return {"preferences": preferences}


def route_to_specialist(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    AGENT 3: Route query to appropriate specialist (music or invoice).
    Depends on: verify_customer_id, load_user_memory.
    
    In a real system, this would invoke LangGraph sub-agents:
    - music_subagent for music queries
    - invoice_subagent for invoice queries
    
    For demo, we simulate specialist responses.
    """
    question = input_data.get("question", "")
    preferences = input_data.get("preferences", "")
    customer_id = input_data.get("customer_id")
    
    # Decide which specialist to route to
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["music", "album", "artist", "track", "song"]):
        # Music specialist
        if "u2" in question_lower:
            response = "We have several U2 albums: Joshua Tree, Achtung Baby, All That You Can't Leave Behind."
        elif "rolling" in question_lower:
            response = "We have albums by Rolling Stones: Sticky Fingers, Exile on Main St., Some Girls."
        else:
            response = f"Based on your preferences ({preferences}), I can help you find music."
    
    elif any(word in question_lower for word in ["invoice", "purchase", "bill", "payment", "recent"]):
        # Invoice specialist
        if customer_id:
            response = f"Your most recent invoice (ID 342) was for $59.99 on 2026-03-10."
        else:
            response = "I need to verify your identity to access invoice information."
    
    else:
        response = "I'm not sure how to help with that. Ask about music, albums, or invoices."
    
    return {"response": response}


def save_updated_preferences(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    AGENT 4: Analyze conversation and save new user preferences.
    Depends on: verify_customer_id, route_to_specialist.
    
    Input: {"customer_id": "123", "question": "...", "response": "..."}
    Output: {"saved": True, "new_preferences": [...]}
    """
    customer_id = input_data.get("customer_id")
    question = input_data.get("question", "")
    response = input_data.get("response", "")
    
    if not customer_id:
        return {"saved": False, "reason": "No customer ID"}
    
    # Extract music preferences from conversation
    new_prefs = extract_music_preferences(question, response)
    
    if new_prefs:
        namespace = ("memory_profile", customer_id)
        # Merge with existing preferences
        existing = memory_store.get(namespace, "user_memory")
        existing_prefs = existing.get("music_preferences", []) if existing else []
        merged_prefs = list(set(existing_prefs + new_prefs))  # Deduplicate
        
        # Save updated profile
        memory_store.put(namespace, "user_memory", {"music_preferences": merged_prefs})
        return {"saved": True, "new_preferences": merged_prefs}
    
    return {"saved": False, "reason": "No new preferences detected"}


# ============================================================================
# BUILD AGENTKUBE-MINI ORCHESTRATION GRAPH
# ============================================================================

def build_orchestration_graph() -> TaskGraph:
    """Construct the orchestration DAG."""
    graph = TaskGraph()
    
    # Add agents with their dependencies
    graph.add(Agent("verify", verify_customer_id))
    graph.add(Agent("load_memory", load_user_memory), depends_on=["verify"])
    graph.add(Agent("route", route_to_specialist), depends_on=["verify", "load_memory"])
    graph.add(Agent("save", save_updated_preferences), depends_on=["verify", "route"])
    
    # Validate the DAG
    graph.validate()
    
    return graph


# ============================================================================
# CONVERSATION LOOP (HANDLES HUMAN-IN-LOOP)
# ============================================================================

def run_conversation(initial_question: str, max_attempts: int = 3) -> str:
    """
    Run a full conversation, handling customer verification if needed.
    
    This is the outer loop that handles human-in-the-loop for verification.
    Agentkube-Mini handles the orchestration inside.
    """
    question = initial_question
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        # Build and run the orchestration graph
        graph = build_orchestration_graph()
        runtime = Runtime(graph)
        result = runtime.run({"question": question})
        
        # Check if verification succeeded
        verify_result = result.memory.get("verify", {})
        customer_id = verify_result.get("customer_id")
        verified = verify_result.get("verified", False)
        
        if verified:
            # Verification succeeded, get the response
            route_result = result.memory.get("route", {})
            response = route_result.get("response", "I'm not sure how to help.")
            
            # Print save result
            save_result = result.memory.get("save", {})
            if save_result.get("saved"):
                print(f"[System] Updated preferences: {save_result.get('new_preferences')}")
            
            return response
        
        else:
            # Verification failed, ask for customer ID
            if attempt < max_attempts:
                print("I need to verify your account first.")
                print("Please provide your Customer ID, email, or phone number.")
                user_input = input("You: ").strip()
                
                if user_input:
                    # Prepend customer ID to question for next attempt
                    question = f"{user_input}. {initial_question}"
                else:
                    print("Verification failed. Please try again later.")
                    return "Account verification unsuccessful."
            else:
                return "Maximum verification attempts exceeded."
    
    return "Session ended."


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    """Run example conversations."""
    
    print("=" * 70)
    print("AGENTKUBE-MINI + LANGGRAPH HYBRID ORCHESTRATION DEMO")
    print("=" * 70)
    
    # Example 1: Query with customer ID provided
    print("\n[EXAMPLE 1] Customer provides ID upfront:")
    print("-" * 70)
    response = run_conversation("My customer ID is 1. What albums by U2 do you have?")
    print(f"Agent: {response}\n")
    
    # Example 2: Query without ID, needs verification
    print("\n[EXAMPLE 2] Customer without ID (simulated):")
    print("-" * 70)
    print("User: What's my most recent invoice?")
    print("Agent: I need to verify your account first.")
    print("Please provide your Customer ID, email, or phone number.")
    # Simulate user input
    response = run_conversation("What's my most recent invoice?")
    # (In interactive mode, this would prompt; for demo, we skip the input loop)
    
    # Example 3: Verify saved preferences
    print("\n[EXAMPLE 3] Saved preferences verification:")
    print("-" * 70)
    namespace = ("memory_profile", "1")
    stored = memory_store.get(namespace, "user_memory")
    print(f"Stored preferences for customer 1: {stored}")
    
    print("\n" + "=" * 70)
    print("ORCHESTRATION GRAPH STRUCTURE:")
    print("=" * 70)
    graph = build_orchestration_graph()
    print(graph.visualize())
    print("\nDAG Mermaid format:")
    print(graph.to_mermaid())


if __name__ == "__main__":
    main()
