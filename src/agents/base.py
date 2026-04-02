import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """State of an agent"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    sender: str
    recipient: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResearchQuery:
    """Structure for research queries"""
    query: str
    topic: str
    num_papers: int = 5
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    """Structure for research results"""
    query: str
    papers: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    insights: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    sources: List[str] = field(default_factory=list)


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.state = AgentState.INITIALIZING
        self.message_history: List[AgentMessage] = []
        self.capabilities = []
    
    def add_capability(self, capability: str):
        """Add a capability to the agent"""
        self.capabilities.append(capability)
    
    def send_message(self, recipient: str, content: str, metadata: Dict[str, Any] = None) -> AgentMessage:
        """Send a message to another agent"""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            metadata=metadata or {}
        )
        self.message_history.append(message)
        return message
    
    def receive_message(self, message: AgentMessage):
        """Receive a message from another agent"""
        self.message_history.append(message)
        logger.info(f"{self.name} received message from {message.sender}: {message.content[:100]}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state"""
        return {
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "capabilities": self.capabilities,
            "message_count": len(self.message_history)
        }


class ResearchAgent(BaseAgent):
    """Agent responsible for retrieving relevant papers"""
    
    def __init__(self, api_client):
        super().__init__(name="ResearchAgent", role="Paper Retrieval")
        self.api_client = api_client
        self.retrieved_papers = []
        self.add_capability("paper_retrieval")
        self.add_capability("web_search")
        self.add_capability("topic_analysis")
    
    async def search_papers(self, query: ResearchQuery) -> List[Dict[str, Any]]:
        """
        Search for relevant papers
        
        Args:
            query: Research query
            
        Returns:
            List of retrieved papers
        """
        self.state = AgentState.PROCESSING
        logger.info(f"Searching for papers: {query.query}")
        
        try:
            # Use chat completion with web_search enabled
            messages = [
                {
                    "role": "user",
                    "content": f"""Find {query.num_papers} academic papers related to:
                    
Topic: {query.topic}
Keywords: {', '.join(query.keywords) if query.keywords else 'general'}
Query: {query.query}

Return the papers in JSON format with fields: title, authors, abstract, year, citations, url."""
                }
            ]
            
            response = await self.api_client.chat_completion(
                messages=messages,
                web_search=True,
                temperature=0.3  # Lower temperature for factual retrieval
            )
            
            # Parse response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract papers (in real scenario, would parse JSON from response)
            papers = self._parse_papers_response(content)
            self.retrieved_papers = papers
            
            self.state = AgentState.COMPLETED
            return papers
            
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Error searching papers: {str(e)}")
            raise
    
    def _parse_papers_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse papers from the API response"""
        # In a real scenario, this would parse JSON
        # For now, return a structured response
        return [
            {
                "title": f"Research Paper {i}",
                "abstract": content[:200],
                "authors": ["AI Research Team"],
                "year": 2024,
                "citations": 0,
                "url": "https://example.com/paper"
            }
            for i in range(1, 4)
        ]


class SummarizationAgent(BaseAgent):
    """Agent responsible for extracting key information"""
    
    def __init__(self, api_client):
        super().__init__(name="SummarizationAgent", role="Key Information Extraction")
        self.api_client = api_client
        self.summaries = {}
        self.add_capability("text_summarization")
        self.add_capability("key_point_extraction")
        self.add_capability("concept_identification")
    
    async def summarize_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize papers and extract key information
        
        Args:
            papers: List of papers to summarize
            
        Returns:
            Dictionary with summaries and key points
        """
        self.state = AgentState.PROCESSING
        logger.info(f"Summarizing {len(papers)} papers")
        
        try:
            summaries = {}
            
            for paper in papers:
                messages = [
                    {
                        "role": "user",
                        "content": f"""Summarize the following academic paper in 200 words, highlighting:
1. Main research question
2. Methodology
3. Key findings
4. Implications

Title: {paper.get('title', 'Unknown')}
Abstract: {paper.get('abstract', '')}"""
                    }
                ]
                
                response = await self.api_client.chat_completion(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=300
                )
                
                summary_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                summaries[paper.get('title', 'Unknown')] = {
                    "summary": summary_content,
                    "key_points": self._extract_key_points(summary_content),
                    "original_abstract": paper.get('abstract', '')
                }
            
            self.summaries = summaries
            self.state = AgentState.COMPLETED
            return summaries
            
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Error summarizing papers: {str(e)}")
            raise
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from summary"""
        # Simple extraction - split by numbers for ordered lists
        points = []
        for line in text.split('\n'):
            if line.strip() and any(line.startswith(str(i)) for i in range(1, 10)):
                points.append(line.strip())
        return points or [text[:100]]


class CriticAgent(BaseAgent):
    """Agent responsible for evaluating information quality"""
    
    def __init__(self, api_client):
        super().__init__(name="CriticAgent", role="Quality Evaluation")
        self.api_client = api_client
        self.evaluations = {}
        self.add_capability("quality_assessment")
        self.add_capability("fact_checking")
        self.add_capability("source_validation")
    
    async def evaluate_information(self, summaries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of summarized information
        
        Args:
            summaries: Dictionary of paper summaries
            
        Returns:
            Quality evaluations and scores
        """
        self.state = AgentState.PROCESSING
        logger.info(f"Evaluating quality of {len(summaries)} summaries")
        
        try:
            evaluations = {}
            
            for title, summary_data in summaries.items():
                messages = [
                    {
                        "role": "user",
                        "content": f"""Evaluate the quality of this research summary on a scale of 1-10:

Title: {title}
Summary: {summary_data.get('summary', '')}

Consider:
1. Clarity and coherence (1-10)
2. Information completeness (1-10)
3. Scientific rigor (1-10)
4. Relevance (1-10)

Provide overall score and brief explanation."""
                    }
                ]
                
                response = await self.api_client.chat_completion(
                    messages=messages,
                    temperature=0.5,
                    max_tokens=200
                )
                
                evaluation_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                score = self._extract_quality_score(evaluation_content)
                evaluations[title] = {
                    "evaluation": evaluation_content,
                    "quality_score": score,
                    "recommendation": "High Quality" if score >= 7 else "Medium Quality" if score >= 5 else "Low Quality"
                }
            
            self.evaluations = evaluations
            self.state = AgentState.COMPLETED
            return evaluations
            
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Error evaluating information: {str(e)}")
            raise
    
    def _extract_quality_score(self, text: str) -> float:
        """Extract quality score from evaluation"""
        # Try to find a number between 1-10
        import re
        scores = re.findall(r'\b([1-9]|10)\b', text)
        if scores:
            return float(sum(int(s) for s in scores) / len(scores))
        return 5.0  # Default score


class WriterAgent(BaseAgent):
    """Agent responsible for compiling findings into a report"""
    
    def __init__(self, api_client):
        super().__init__(name="WriterAgent", role="Report Generation")
        self.api_client = api_client
        self.reports = {}
        self.add_capability("report_generation")
        self.add_capability("content_creation")
        self.add_capability("formatting")
    
    async def generate_report(
        self,
        query: str,
        summaries: Dict[str, Any],
        evaluations: Dict[str, Any],
        research_result: Optional[ResearchResult] = None
    ) -> str:
        """
        Generate comprehensive research report
        
        Args:
            query: Original research query
            summaries: Paper summaries
            evaluations: Quality evaluations
            research_result: Optional research result object
            
        Returns:
            Generated report
        """
        self.state = AgentState.PROCESSING
        logger.info(f"Generating report for query: {query}")
        
        try:
            # Prepare context
            summary_text = "\n\n".join([
                f"Paper: {title}\n{data['summary']}"
                for title, data in summaries.items()
            ])
            
            evaluation_text = "\n\n".join([
                f"Paper: {title}\n Quality Score: {data['quality_score']}/10\n{data['recommendation']}"
                for title, data in evaluations.items()
            ])
            
            messages = [
                {
                    "role": "user",
                    "content": f"""Generate a comprehensive research report based on the following:

Research Query: {query}

PAPER SUMMARIES:
{summary_text}

QUALITY EVALUATIONS:
{evaluation_text}

Create a well-structured report with:
1. Executive Summary
2. Literature Review
3. Key Findings
4. Gaps in Current Research
5. Recommendations for Future Work
6. Conclusion

Format as markdown with proper sections."""
                }
            ]
            
            response = await self.api_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            report = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if research_result:
                research_result.summary = report
            
            self.reports[query] = report
            self.state = AgentState.COMPLETED
            return report
            
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Error generating report: {str(e)}")
            raise
