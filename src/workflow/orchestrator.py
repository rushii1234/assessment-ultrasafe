import logging
from typing import Dict, Any, List, Annotated
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from src.agents.base import (
    ResearchAgent, SummarizationAgent, CriticAgent, WriterAgent,
    ResearchQuery, ResearchResult, AgentMessage
)
from src.rag.retrieval import DocumentChunker, HybridRetriever, ContextualCompressor, CrossDocumentSynthesizer
from src.api.client import api_client

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """States in the research workflow"""
    INITIALIZED = "initialized"
    SEARCHING = "searching"
    SUMMARIZING = "summarizing"
    EVALUATING = "evaluating"
    WRITING = "writing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class WorkflowInput:
    """Input for the research workflow"""
    query: str
    topic: str
    keywords: List[str] = field(default_factory=list)
    num_papers: int = 5
    enable_rag: bool = True
    enable_reranking: bool = True


@dataclass
class WorkflowContext:
    """Context passed through the workflow"""
    state: WorkflowState = WorkflowState.INITIALIZED
    input: WorkflowInput = None
    research_query: ResearchQuery = None
    retrieved_papers: List[Dict[str, Any]] = field(default_factory=list)
    summaries: Dict[str, Any] = field(default_factory=dict)
    evaluations: Dict[str, Any] = field(default_factory=dict)
    final_report: str = ""
    errors: List[str] = field(default_factory=list)
    agent_messages: List[AgentMessage] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add error to context"""
        self.errors.append(error)
        logger.error(error)


class ResearchOrchestrator:
    """Orchestrates the multi-agent research workflow using LangGraph approach"""
    
    def __init__(self):
        self.research_agent = ResearchAgent(api_client)
        self.summarization_agent = SummarizationAgent(api_client)
        self.critic_agent = CriticAgent(api_client)
        self.writer_agent = WriterAgent(api_client)
        
        # RAG components
        self.chunker = DocumentChunker()
        self.retriever = HybridRetriever()
        self.compressor = ContextualCompressor()
        self.synthesizer = CrossDocumentSynthesizer()
        
        # Store executed workflows
        self.workflow_history = []
    
    async def execute_workflow(self, workflow_input: WorkflowInput) -> ResearchResult:
        """
        Execute the complete research workflow
        
        Args:
            workflow_input: Input parameters for the workflow
            
        Returns:
            Research result
        """
        context = WorkflowContext(input=workflow_input)
        
        try:
            # Step 1: Initialize Research Query
            context = await self._initialize(context)
            
            # Step 2: Research (Retrieve Papers)
            context = await self._search_papers(context)
            
            # Step 3: Summarization
            context = await self._summarize_papers(context)
            
            # Step 4: Evaluation
            context = await self._evaluate_quality(context)
            
            # Step 5: Writing
            context = await self._generate_report(context)
            
            # Step 6: Synthesis and Insights
            context = await self._synthesize_insights(context)
            
            context.state = WorkflowState.COMPLETED
            
        except Exception as e:
            context.state = WorkflowState.ERROR
            context.add_error(f"Workflow execution failed: {str(e)}")
        
        # Store in history
        self.workflow_history.append(context)
        
        # Prepare result
        result = self._prepare_result(context)
        return result
    
    async def _initialize(self, context: WorkflowContext) -> WorkflowContext:
        """Initialize the workflow"""
        logger.info("Initializing research workflow")
        
        context.research_query = ResearchQuery(
            query=context.input.query,
            topic=context.input.topic,
            num_papers=context.input.num_papers,
            keywords=context.input.keywords
        )
        
        return context
    
    async def _search_papers(self, context: WorkflowContext) -> WorkflowContext:
        """Search for relevant papers"""
        context.state = WorkflowState.SEARCHING
        logger.info(f"Searching for papers: {context.research_query.query}")
        
        try:
            papers = await self.research_agent.search_papers(context.research_query)
            context.retrieved_papers = papers
            
            # Send message to next agent
            message = self.research_agent.send_message(
                recipient=self.summarization_agent.name,
                content=f"Retrieved {len(papers)} papers for {context.research_query.query}",
                metadata={"papers": len(papers)}
            )
            context.agent_messages.append(message)
            
        except Exception as e:
            context.add_error(f"Paper search failed: {str(e)}")
        
        return context
    
    async def _summarize_papers(self, context: WorkflowContext) -> WorkflowContext:
        """Summarize retrieved papers"""
        context.state = WorkflowState.SUMMARIZING
        logger.info("Summarizing papers")
        
        try:
            if not context.retrieved_papers:
                logger.warning("No papers to summarize")
                return context
            
            summaries = await self.summarization_agent.summarize_papers(context.retrieved_papers)
            context.summaries = summaries
            
            message = self.summarization_agent.send_message(
                recipient=self.critic_agent.name,
                content=f"Summarized {len(summaries)} papers",
                metadata={"summaries": len(summaries)}
            )
            context.agent_messages.append(message)
            
        except Exception as e:
            context.add_error(f"Paper summarization failed: {str(e)}")
        
        return context
    
    async def _evaluate_quality(self, context: WorkflowContext) -> WorkflowContext:
        """Evaluate quality of summarized information"""
        context.state = WorkflowState.EVALUATING
        logger.info("Evaluating information quality")
        
        try:
            if not context.summaries:
                logger.warning("No summaries to evaluate")
                return context
            
            evaluations = await self.critic_agent.evaluate_information(context.summaries)
            context.evaluations = evaluations
            
            message = self.critic_agent.send_message(
                recipient=self.writer_agent.name,
                content=f"Evaluated {len(evaluations)} summaries",
                metadata={"evaluations": len(evaluations)}
            )
            context.agent_messages.append(message)
            
        except Exception as e:
            context.add_error(f"Quality evaluation failed: {str(e)}")
        
        return context
    
    async def _generate_report(self, context: WorkflowContext) -> WorkflowContext:
        """Generate comprehensive report"""
        context.state = WorkflowState.WRITING
        logger.info("Generating research report")
        
        try:
            research_result = ResearchResult(
                query=context.input.query,
                papers=context.retrieved_papers
            )
            
            report = await self.writer_agent.generate_report(
                query=context.input.query,
                summaries=context.summaries,
                evaluations=context.evaluations,
                research_result=research_result
            )
            
            context.final_report = report
            
            message = self.writer_agent.send_message(
                recipient="Orchestrator",
                content="Report generation completed",
                metadata={"report_length": len(report)}
            )
            context.agent_messages.append(message)
            
        except Exception as e:
            context.add_error(f"Report generation failed: {str(e)}")
        
        return context
    
    async def _synthesize_insights(self, context: WorkflowContext) -> WorkflowContext:
        """Synthesize cross-document insights"""
        logger.info("Synthesizing insights")
        
        try:
            if not context.retrieved_papers:
                return context
            
            # Extract documents from papers
            documents = [p.get('abstract', '') for p in context.retrieved_papers]
            documents = [d for d in documents if d]  # Filter empty
            
            # Perform synthesis
            synthesis = self.synthesizer.synthesize_findings(documents, context.input.query)
            
            # Add insights to context
            context = self._extract_insights(context)
            
        except Exception as e:
            context.add_error(f"Insight synthesis failed: {str(e)}")
        
        return context
    
    def _extract_insights(self, context: WorkflowContext) -> WorkflowContext:
        """Extract insights from evaluations"""
        insights = []
        
        for title, eval_data in context.evaluations.items():
            if eval_data.get('quality_score', 0) >= 7:
                insights.append(f"High-quality findings from: {title}")
        
        # Also add synthesis insights
        if context.retrieved_papers:
            insights.append(f"Analyzed {len(context.retrieved_papers)} papers for {context.input.query}")
        
        return context
    
    def _prepare_result(self, context: WorkflowContext) -> ResearchResult:
        """Prepare final research result"""
        result = ResearchResult(
            query=context.input.query,
            papers=context.retrieved_papers,
            summary=context.final_report,
            quality_score=self._calculate_quality_score(context)
        )
        
        # Extract insights
        for title, eval_data in context.evaluations.items():
            if eval_data.get('quality_score', 0) >= 7:
                result.insights.append(
                    f"{title}: {eval_data.get('recommendation', 'No recommendation')}"
                )
        
        return result
    
    def _calculate_quality_score(self, context: WorkflowContext) -> float:
        """Calculate overall quality score"""
        if not context.evaluations:
            return 0.0
        
        scores = [
            eval_data.get('quality_score', 0)
            for eval_data in context.evaluations.values()
        ]
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get statistics about executed workflows"""
        if not self.workflow_history:
            return {"workflows_executed": 0}
        
        completed = sum(1 for w in self.workflow_history if w.state == WorkflowState.COMPLETED)
        errors = sum(1 for w in self.workflow_history if w.state == WorkflowState.ERROR)
        total_papers = sum(len(w.retrieved_papers) for w in self.workflow_history)
        
        return {
            "workflows_executed": len(self.workflow_history),
            "completed": completed,
            "errors": errors,
            "total_papers_retrieved": total_papers,
            "avg_papers_per_workflow": total_papers / len(self.workflow_history) if self.workflow_history else 0
        }


# Singleton instance
orchestrator = ResearchOrchestrator()
