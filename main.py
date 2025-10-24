#!/usr/bin/env python3
"""
Main entry point for the ModelCA League Simulator.
"""
import sys
import os

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run the ModelCA League Simulator."""
    print("ModelCA League Simulator")
    print("========================")
    print("1. Run Streamlit UI (step-by-step)")
    print("2. Run League Generator")
    print("3. Run Season Simulator")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Run the Streamlit UI
        import streamlit_step_by_step
        # Note: Streamlit apps are typically run from the command line
        print("To run the Streamlit UI, use:")
        print("streamlit run streamlit_step_by_step.py")
        
    elif choice == "2":
        # Run the league generator
        import make_league
        make_league.main()
        
    elif choice == "3":
        # Run the season simulator
        import simulate_seasons_roster_style
        simulate_seasons_roster_style.main()
        
    elif choice == "4":
        print("Goodbye!")
        sys.exit(0)
        
    else:
        print("Invalid choice. Please run again.")
        sys.exit(1)

if __name__ == "__main__":
    main()