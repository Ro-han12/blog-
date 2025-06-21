from crewai import Crew

class ResearchCrews:
    """Factory for creating crews used in the research conversion process."""
    
    @staticmethod
    def create_research_to_content_crew(agents, tasks):
        """Create a crew for the research to content pipeline.
        
        Args:
            agents: List of agents in the crew
            tasks: List of tasks to perform
            
        Returns:
            Configured Crew instance
        """
        return Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        ) 